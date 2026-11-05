#include "MiciusShell.h"
#include "gguf/FileReader.h"
#include "text/Conversation.h"
#include <charconv>
#include <iostream>
#include <stdexcept>

namespace {
uint32_t Number(const std::string& value)
{
    uint32_t n=0;
    auto result=std::from_chars(value.data(),value.data()+value.size(),n);
    if (result.ec!=std::errc{} || result.ptr!=value.data()+value.size() || !n)
        throw std::invalid_argument("Expected a positive integer");
    return n;
}

// A byte-BPE token may end inside a codepoint. Only emit complete UTF-8;
// generated C0/C1 controls never become terminal escape commands.
class Display {
    MiciusShell& shell;
    std::string pending;
public:
    explicit Display(MiciusShell& s) : shell(s) {}
    void Write(std::string_view bytes, bool final=false)
    {
        pending.append(bytes);
        std::string out;
        size_t i=0;
        while (i<pending.size()) {
            const auto lead=static_cast<unsigned char>(pending[i]);
            const size_t len=lead<128 ? 1 : lead>=0xc2 && lead<=0xdf ? 2 :
                lead>=0xe0 && lead<=0xef ? 3 : lead>=0xf0 && lead<=0xf4 ? 4 : 0;
            bool valid=len!=0;
            uint32_t cp=lead & (len==1 ? 127u : len==2 ? 31u : len==3 ? 15u : 7u);
            size_t j=1;
            for (; valid && j<len && j<pending.size()-i; ++j) {
                const auto c=static_cast<unsigned char>(pending[i+j]);
                valid=(c&0xc0)==0x80 && (j!=1 ||
                    !((lead==0xe0 && c<0xa0) || (lead==0xed && c>=0xa0) ||
                      (lead==0xf0 && c<0x90) || (lead==0xf4 && c>=0x90)));
                cp=(cp<<6)|(c&63u);
            }
            if (valid && j<len && !final) break;
            if (!valid || j<len) {
                out+="\xef\xbf\xbd"; i+=valid ? j : 1; continue;
            }
            if ((cp<32 && cp!='\n' && cp!='\t') || (cp>=127 && cp<=159)) out+="\xef\xbf\xbd";
            else out.append(pending,i,len);
            i+=len;
        }
        pending.erase(0,i);
        if (!out.empty()) { shell.ShellPrint(out); std::cout.flush(); }
    }
};
}

int MiciusShell::ChatLoop(uint32_t max_new, bool trace)
{
    if (!ctx.file_model) throw std::invalid_argument("Load a GGUF model before entering chat");
    using namespace micius::text;
    Tokenizer tokenizer(micius::gguf::FileReader(*ctx.file_model));
    Conversation session(ctx.engine_ctx,ctx.engine_config.max_seq_len,tokenizer);
    ShellPrint("[Chat] UTF-8 chat. :reset starts over, :log start/stop controls the transcript, :exit leaves chat.\n");
    size_t turns=0;
    for (;;) {
        ctx.is_interrupted.store(false,std::memory_order_relaxed);
        std::cout<<"\033[36mYou: \033[0m"<<std::flush;
        std::string line;
        if (!ReadLine(line)) {
            if (!std::cin.eof() && ctx.is_interrupted.load()) { std::cin.clear(); continue; }
            break;
        }
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (line.empty()) continue;
        LogInput("You: ",line);
        if (line==":exit") break;
        if (line==":log" || line.starts_with(":log ") || line.starts_with(":log\t")) {
            try { CmdLog(this,&ctx,TokenizeInput(line.substr(1))); }
            catch (const std::exception& e) { ShellPrint(std::string("[Log] ")+e.what()+"\n"); }
            continue;
        }
        if (line==":reset") {
            session.Reset(); turns=0;
            ShellPrint("[Chat] Conversation reset.\n");
            if (trace) std::cerr<<"[chat-reset]\n";
            continue;
        }
        Display display(*this);
        ShellPrint("\033[32mAssistant: \033[0m"); std::cout.flush();
        try {
            const auto answer=session.Reply(line,max_new,ctx.is_interrupted,
                [&](std::string_view piece) { display.Write(piece); });
            display.Write({},true);
            ShellPrint("\n");
            const char* reason="eot";
            if (answer.reason==Finish::Interrupted) {
                reason="interrupted";
                ShellPrint("[Chat] Interrupted. This turn was discarded.\n");
            } else {
                ++turns;
                if (answer.reason!=Finish::End) {
                    reason=answer.reason==Finish::ContextLimit ? "context-limit" : "reply-limit";
                    ShellPrint(std::string("[Chat] ")+reason+" reached.\n");
                }
            }
            if (trace) std::cerr<<"[chat-turn] turn="<<turns<<" finish="<<reason<<" tokens="<<answer.tokens
                                <<" context="<<Micius_GetSequenceLengthV1(ctx.engine_ctx)<<"\n";
        } catch (const std::exception& e) {
            display.Write({},true);
            ShellPrint(std::string("\n[Chat] Turn rejected: ")+e.what()+"\n");
            if (trace) std::cerr<<"[chat-error]\n";
        }
    }
    session.Reset();
    return 0;
}

void MiciusShell::CmdChat(MiciusShell* shell, ShellContext*, const std::vector<std::string>& args)
{
    if (args.size()>2) throw std::invalid_argument("Usage: chat [max_reply_tokens]");
    shell->ChatLoop(args.size()==2 ? Number(args[1]) : 128);
}

int MiciusShell::RunChat(const std::vector<std::string>& args)
{
    if (args.empty() || args[0].empty()) throw std::invalid_argument("Usage: --chat MODEL [--context N] [--max-new N]");
    uint32_t context=1024, reply=128;
    bool trace=false;
    for (size_t i=1;i<args.size();++i) {
        const auto& option=args[i];
        if (option=="--trace-turns") { trace=true; continue; }
        if ((option!="--context" && option!="--max-new") || i+1==args.size())
            throw std::invalid_argument("Invalid --chat option: "+option);
        (option=="--context" ? context : reply)=Number(args[++i]);
    }
    return LoadFile(args[0],context) ? ChatLoop(reply,trace) : 1;
}
