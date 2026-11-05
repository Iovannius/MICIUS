#include "Tokenizer.h"
#include "app/MiciusGGUFParser.h"
#include "gguf/ContextDirectory.h"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>

namespace micius::text {
namespace {
constexpr uint32_t missing = UINT32_MAX;
uint64_t Pair(uint32_t left, uint32_t right) { return (uint64_t(left) << 32) | right; }
void Require(bool valid, const char* why) { if (!valid) throw std::runtime_error(why); }

template<class T> T Take(std::string_view& span)
{
    Require(span.size() >= sizeof(T), "Truncated tokenizer metadata");
    T v;
    std::memcpy(&v,span.data(),sizeof(v));
    span.remove_prefix(sizeof(v));
    return v;
}
std::string_view TakeString(std::string_view& span)
{
    const auto n = Take<uint64_t>(span);
    Require(n <= span.size(), "Truncated BPE merge");
    const auto s = span.substr(0,static_cast<size_t>(n));
    span.remove_prefix(s.size());
    return s;
}

class Pattern {
    std::unique_ptr<pcre2_code,decltype(&pcre2_code_free)> code{nullptr,pcre2_code_free};
public:
    explicit Pattern(const char* pattern)
    {
        PCRE2_SIZE offset = 0; int err = 0;
        code.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern),PCRE2_ZERO_TERMINATED,
                                 PCRE2_UTF|PCRE2_UCP,&err,&offset,nullptr));
        Require(bool(code),"Unicode pre-tokenizer could not be compiled");
    }
    template<class F> void Each(std::string_view s, F emit) const
    {
        if (s.empty()) return;
        std::unique_ptr<pcre2_match_data,decltype(&pcre2_match_data_free)> match(
            pcre2_match_data_create_from_pattern(code.get(),nullptr),pcre2_match_data_free);
        if (!match) throw std::bad_alloc();
        for (size_t pos=0; pos<s.size();) {
            // The first complete-subject match validates UTF-8. All later offsets
            // come from codepoint-aligned matches on those same immutable bytes.
            const int rc = pcre2_match(code.get(),reinterpret_cast<PCRE2_SPTR>(s.data()),s.size(),pos,
                PCRE2_ANCHORED | (pos ? PCRE2_NO_UTF_CHECK : 0u),match.get(),nullptr);
            if (rc < 1) throw std::invalid_argument("Invalid UTF-8 or BPE pre-tokenization limit exceeded");
            const auto* ends = pcre2_get_ovector_pointer(match.get());
            Require(ends[0] == pos && ends[1] > pos,"Invalid pre-tokenizer boundary");
            emit(s.substr(pos,ends[1]-pos),rc);
            pos = ends[1];
        }
    }
};

// The GGUF profiles split ordinary text before byte-BPE. SmolLM first isolates
// each Unicode number; spaces ending the preceding fragment matter here.
const Pattern& QwenPattern()
{
    static const Pattern p(R"re((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re"); return p;
}
const Pattern& SmolPattern()
{
    static const Pattern p(R"re('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)re"); return p;
}
const Pattern& NumberPattern()
{
    static const Pattern p(R"re((\p{N})|[^\p{N}]+)re"); return p;
}

std::array<int,512> ByteAlphabet()
{
    std::array<int,512> reverse;
    reverse.fill(-1);
    size_t substituted = 256;
    for (int b=0; b<256; ++b) {
        const bool direct = (b>=33 && b<=126) || (b>=161 && b<=172) || b>=174;
        reverse[direct ? static_cast<size_t>(b) : substituted++] = b;
    }
    return reverse;
}
std::string FromAlphabet(std::string_view token, const std::array<int,512>& reverse)
{
    std::string bytes;
    for (size_t i=0; i<token.size();) {
        unsigned c=static_cast<unsigned char>(token[i++]);
        if (c>=128) {
            Require(c>=0xc2 && c<=0xc7 && i<token.size(),"Character outside BPE byte alphabet");
            const auto second=static_cast<unsigned char>(token[i++]);
            Require((second&0xc0)==0x80,"Invalid BPE alphabet encoding");
            c=((c&31u)<<6)|(second&63u);
        }
        Require(c<reverse.size() && reverse[c]>=0,"Invalid BPE alphabet character");
        bytes.push_back(static_cast<char>(reverse[c]));
    }
    return bytes;
}
}

Tokenizer::Tokenizer(const GGUFParser& reader)
{
    const auto pre = reader.TextString("tokenizer.ggml.pre");
    Require(reader.TextString("tokenizer.ggml.model")=="gpt2" && (pre=="smollm" || pre=="qwen2"),
            "Text requires GGUF byte-BPE with smollm or qwen2 preprocessing");
    for (auto key : {"tokenizer.ggml.add_bos_token","tokenizer.ggml.add_eos_token","tokenizer.ggml.add_space_prefix"})
        Require(!reader.TextFlag(key),"Unsupported implicit special token or prefix space");
    qwen = pre=="qwen2";
    const auto& words = reader.GetVocab();
    auto types = reader.TypesRecord();
    Require(Take<uint32_t>(types)==5 && Take<uint64_t>(types)==words.size() &&
            words.size()<missing && types.size()/4==words.size(),"Expected one INT32 token type per token");
    const auto reverse=ByteAlphabet();
    std::unordered_map<std::string_view,uint32_t> ids;
    ids.reserve(words.size());
    std::vector<int32_t> kinds(words.size());
    pieces.resize(words.size()); bytes.fill(missing);
    for (uint32_t i=0; i<words.size(); ++i) {
        const auto type=kinds[i]=Take<int32_t>(types);
        Require(!words[i].empty() && ids.emplace(words[i],i).second,"Empty or duplicate BPE vocabulary entry");
        if (type==1) {
            pieces[i]=FromAlphabet(words[i],reverse);
            if (pieces[i].size()==1) bytes[static_cast<unsigned char>(pieces[i][0])]=i;
        } else if (type==4) {
            static const Pattern valid(R"((?s).+)");
            valid.Each(words[i],[](std::string_view,int){});
            pieces[i]=words[i]; user_defined.push_back(i);
        } else {
            Require(type==2 || type==3 || type==5,"Unsupported byte-BPE token type");
        }
    }
    // SmolLM2's published vocabulary omits 21 single-byte symbols. Preserve
    // ordinary input exactly and reject an unavailable byte only when used.
    Require(std::find(bytes.begin()+32,bytes.begin()+127,missing)==bytes.begin()+127 &&
            bytes['\n']!=missing && bytes['\r']!=missing && bytes['\t']!=missing,
            "Incomplete basic text alphabet in the GGUF vocabulary");
    auto control=[&](const char* word) {
        const auto it=ids.find(word);
        Require(it!=ids.end() && kinds[it->second]==3,"Missing ChatML control token");
        return it->second;
    };
    start=control("<|im_start|>"); end=control("<|im_end|>"); end_of_text=control("<|endoftext|>");
    Require(reader.TextUint("tokenizer.ggml.eos_token_id")==end,"Unsupported chat EOS");
    auto ranks=reader.MergesRecord();
    Require(Take<uint32_t>(ranks)==8,"Expected an array of BPE merge strings");
    const uint64_t count=Take<uint64_t>(ranks);
    Require(count<missing && count<=ranks.size()/8,"Invalid BPE merge count");
    merges.reserve(static_cast<size_t>(count));
    for (uint32_t i=0;i<count;++i) {
        const auto rule=TakeString(ranks); const size_t split=rule.find(' ');
        Require(split!=rule.npos && split && split+1<rule.size(),"Malformed BPE merge");
        auto l=ids.find(rule.substr(0,split)), r=ids.find(rule.substr(split+1));
        std::string merged(rule.substr(0,split)); merged.append(rule.substr(split+1));
        auto v=ids.find(merged);
        Require(l!=ids.end() && r!=ids.end() && v!=ids.end() && kinds[l->second]==1 &&
                kinds[r->second]==1 && kinds[v->second]==1,"BPE merge refers to a missing/non-ordinary token");
        Require(merges.emplace(Pair(l->second,r->second),Merge{i,v->second}).second,"Duplicate BPE merge");
    }
    Require(ranks.empty() && types.empty(),"Trailing bytes in tokenizer metadata");
    const auto tmpl=reader.TextString("tokenizer.chat_template");
    if (!tmpl.empty()) {
        const auto hash=gguf::Sha256(tmpl.data(),tmpl.size());
        if (!qwen && hash=="872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788")
            system_message="You are a helpful AI assistant named SmolLM, trained by Hugging Face";
        if (qwen && hash=="d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4")
            system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    }
}

void Tokenizer::MergePiece(std::string_view piece, std::vector<uint32_t>& out) const
{
    auto lookup=[&](unsigned char b) {
        if (bytes[b]==missing) throw std::invalid_argument("Input contains a byte absent from this GGUF vocabulary");
        return bytes[b];
    };
    if (piece.size()==1) { out.push_back(lookup(static_cast<unsigned char>(piece[0]))); return; }
    struct Node { uint32_t id, prev, next; };
    std::vector<Node> nodes;
    nodes.reserve(piece.size());
    for (unsigned char byte : piece) {
        const auto i=static_cast<uint32_t>(nodes.size());
        nodes.push_back({lookup(byte),i==0 ? missing : i-1,i+1});
    }
    nodes.back().next=missing;
    using Event=std::pair<uint32_t,uint32_t>; // lowest merge rank, then leftmost position
    std::priority_queue<Event,std::vector<Event>,std::greater<Event>> queue;
    auto offer=[&](uint32_t left) {
        if (left==missing || nodes[left].next==missing) return;
        auto it=merges.find(Pair(nodes[left].id,nodes[nodes[left].next].id));
        if (it!=merges.end()) queue.emplace(it->second.rank,left);
    };
    for (uint32_t i=0;i+1<nodes.size();++i) offer(i);
    while (!queue.empty()) {
        const auto [rank,left]=queue.top(); queue.pop();
        const auto right=nodes[left].next;
        if (right==missing || nodes[left].id==missing) continue;
        const auto it=merges.find(Pair(nodes[left].id,nodes[right].id));
        if (it==merges.end() || it->second.rank!=rank) continue;
        nodes[left].id=it->second.result;
        nodes[right].id=missing;
        nodes[left].next=nodes[right].next;
        if (nodes[left].next!=missing) nodes[nodes[left].next].prev=left;
        offer(nodes[left].prev); offer(left);
    }
    for (uint32_t i=0;i!=missing;i=nodes[i].next) out.push_back(nodes[i].id);
}

std::vector<uint32_t> Tokenizer::Encode(std::string_view text) const
{
    if (text.size()>16u*1024*1024) throw std::length_error("Text exceeds 16 MiB per call");
    std::vector<uint32_t> out;
    auto emit=[&](std::string_view p,int) { MergePiece(p,out); };
    while (!text.empty()) {
        size_t boundary=text.size(); uint32_t matched=missing;
        for (uint32_t id:user_defined) {
            const size_t pos=text.find(pieces[id]);
            if (pos<boundary || (pos==boundary && matched!=missing && pieces[id].size()>pieces[matched].size())) {
                boundary=pos; matched=id;
            }
        }
        const auto normal=text.substr(0,boundary);
        if (qwen) QwenPattern().Each(normal,emit);
        else NumberPattern().Each(normal,[&](std::string_view s,int captured) {
            if (captured==2) MergePiece(s,out); else SmolPattern().Each(s,emit);
        });
        if (matched==missing) break;
        out.push_back(matched);
        text.remove_prefix(boundary+pieces[matched].size());
    }
    return out;
}

std::string_view Tokenizer::Piece(uint32_t id) const
{
    if (id>=pieces.size() || pieces[id].empty()) throw std::invalid_argument("Non-text output token");
    return pieces[id];
}

std::vector<uint32_t> Tokenizer::EndTurn() const
{
    auto ids=Encode("\n");
    ids.insert(ids.begin(),end);
    return ids;
}

std::vector<uint32_t> Tokenizer::Prompt(bool first, std::string_view user) const
{
    if (!SupportsChat()) throw std::runtime_error("Unrecognized GGUF chat template");
    std::vector<uint32_t> ids;
    auto append=[&](const std::vector<uint32_t>& v) { ids.insert(ids.end(),v.begin(),v.end()); };
    auto message=[&](const char* role,std::string_view body) {
        ids.push_back(start);
        // Encode the role + newline and content together. Splitting them can
        // change BPE boundaries for whitespace at the start of the message.
        std::string payload(role); payload.push_back('\n'); payload.append(body);
        append(Encode(payload)); append(EndTurn());
    };
    if (first) message("system",system_message);
    message("user",user);
    ids.push_back(start); append(Encode("assistant\n"));
    return ids;
}
}
