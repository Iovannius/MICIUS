#include "I4Evaluation.h"
#include "AI/I4MixedPrecision.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace micius::i4 {
namespace {
uint32_t ReadU32(std::istream& stream) {
    uint32_t value=0;stream.read(reinterpret_cast<char*>(&value),4);
    if(!stream)throw std::runtime_error("truncated token input");
    return value;
}
struct Sum {
    double sum=0,correction=0;
    void add(double v) noexcept {
        const double next=sum+v;
        correction+=std::abs(sum)>=std::abs(v) ? (sum-next)+v : (v-next)+sum;
        sum=next;
    }
    double value() const noexcept{return sum+correction;}
};
}
bool ScoreLogits(std::span<const float> logits,uint32_t target,Score& score) noexcept {
    if(logits.empty() || target>=logits.size())return false;
    double maximum=-std::numeric_limits<double>::infinity();uint32_t best=0;
    for(size_t i=0;i<logits.size();++i) {
        if(!std::isfinite(logits[i]))return false;
        if(logits[i]>maximum){maximum=logits[i];best=static_cast<uint32_t>(i);}
    }
    Sum sum;
    for(float x:logits)sum.add(std::exp(static_cast<double>(x)-maximum));
    const double log_sum=std::log(sum.value());
    score={maximum-static_cast<double>(logits[target])+log_sum,maximum+log_sum,logits[target],best};
    return std::isfinite(score.nll) && score.nll>=0 && std::isfinite(score.logsumexp);
}
std::vector<std::vector<uint32_t>> ReadTokens(const std::string& path,uint32_t vocab,uint32_t context) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    std::ifstream stream(path,std::ios::binary);
    char magic[8]{};stream.read(magic,8);
    if(!stream || std::memcmp(magic,"MICI4T1\0",8)!=0)throw std::runtime_error("invalid token file magic");
    if(ReadU32(stream)!=vocab)throw std::runtime_error("token vocabulary mismatch");
    const auto count=ReadU32(stream);
    if(!count || count>128)throw std::runtime_error("invalid segment count");
    std::vector<std::vector<uint32_t>> segments;segments.reserve(count);
    for(uint32_t i=0;i<count;++i) {
        const auto size=ReadU32(stream);
        if(size<2 || static_cast<uint64_t>(size)>static_cast<uint64_t>(context)+1 || size>1048576)
            throw std::runtime_error("segment exceeds context or has no target");
        auto& segment=segments.emplace_back();segment.reserve(size);
        for(uint32_t j=0;j<size;++j) {
            const auto token=ReadU32(stream);
            if(token>=vocab)throw std::runtime_error("token outside vocabulary");
            segment.push_back(token);
        }
    }
    if(stream.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing token payload");
    return segments;
}
int EvaluateTokens(MiciusContext* ctx,const MiciusConfig& config,const std::string& input,const std::string& output) noexcept {
    const auto fail=[](const char* what){std::cout<<"status=FAIL\nerror="<<what<<'\n';return 1;};
    try {
        if(!ctx || (_mm_getcsr()&0xffc0U)!=0x1f80U)return fail("invalid evaluation context or FP control");
        if((CompiledPolicy()==Policy::L31GU || CompiledPolicy()==Policy::BothFFN) && config.num_layers<32)
            return fail("selected FFN layer absent");
        auto segments=ReadTokens(input,config.vocab_size,config.max_seq_len);
        if(!std::filesystem::create_directory(output))return fail("evaluation requires a new output directory");
        const std::filesystem::path root(output);
        std::ofstream records(root/"scores.jsonl",std::ios::binary);
        std::ofstream tensors(root/"logits.f32",std::ios::binary);
        if(!records || !tensors)return fail("cannot create evaluation files");
        records<<std::setprecision(17)<<"{\"type\":\"begin\",\"schema\":1,\"vocab\":"<<config.vocab_size
            <<",\"segments\":"<<segments.size()<<",\"layers\":"<<config.num_layers<<",\"policy\":\""<<PolicyName(CompiledPolicy())
            <<"\",\"i1_mode\":"<<MICIUS_I1_MODE<<",\"mxcsr\":"<<_mm_getcsr()<<",\"abi\":"<<MICIUS_ABI_VERSION<<"}\n";
        std::vector<float> logits(config.vocab_size);
        uint64_t count=0;Sum all;
        for(size_t segment=0;segment<segments.size();++segment) {
            if(!Micius_ResetEngineV1(ctx) || Micius_GetSequenceLengthV1(ctx)!=0)return fail("reset failed");
            const auto& ids=segments[segment];Sum sum;
            for(size_t pos=0;pos+1<ids.size();++pos) {
                RouteAudit audit(config.num_layers);
                if(!audit.good() || Micius_PrefillTokenV1(ctx,ids[pos])!=MICIUS_STATE_TOKEN_ACCEPTED ||
                   Micius_GetSequenceLengthV1(ctx)!=pos+1 || !audit.finish() ||
                   !Micius_CopyLastLogitsV1(ctx,logits.data(),logits.size()))return fail("teacher-forced step or route audit failed");
                Score score;
                if(!ScoreLogits(logits,ids[pos+1],score) || (_mm_getcsr()&0xffc0U)!=0x1f80U)return fail("invalid logits or FP control");
                tensors.write(reinterpret_cast<const char*>(logits.data()),static_cast<std::streamsize>(logits.size()*4));
                records<<"{\"type\":\"token\",\"segment\":"<<segment<<",\"position\":"<<pos
                    <<",\"input\":"<<ids[pos]<<",\"target\":"<<ids[pos+1]<<",\"top1\":"<<score.top1
                    <<",\"nll\":"<<score.nll<<",\"logsumexp\":"<<score.logsumexp<<",\"target_logit\":"<<score.target_logit
                    <<",\"f64_override_calls\":"<<audit.count()<<",\"logits_offset\":"<<count*config.vocab_size*4ULL
                    <<",\"mxcsr\":"<<_mm_getcsr()<<"}\n";
                if(!records || !tensors)return fail("evaluation write failed");
                sum.add(score.nll);all.add(score.nll);++count;
            }
            records<<"{\"type\":\"segment_complete\",\"segment\":"<<segment<<",\"count\":"<<ids.size()-1
                   <<",\"nll_sum\":"<<sum.value()<<"}\n";
            std::cout<<"i4_segment="<<segment<<" scored="<<ids.size()-1<<"\n"<<std::flush;
        }
        const double mean=all.value()/static_cast<double>(count),ppl=std::exp(mean);
        if(!std::isfinite(ppl))return fail("perplexity overflow");
        records<<"{\"type\":\"complete\",\"count\":"<<count<<",\"nll_sum\":"<<all.value()
               <<",\"mean_nll\":"<<mean<<",\"perplexity\":"<<ppl<<",\"logits_bytes\":"<<count*config.vocab_size*4ULL<<"}\n";
        records.flush();tensors.flush();
        if(!records || !tensors)return fail("flush failed");
        records.close();tensors.close();
        if(!records || !tensors)return fail("close failed");
        std::cout<<"mode=i4_teacher_forced_nll\npolicy="<<PolicyName(CompiledPolicy())
                 <<"\nscored_tokens="<<count<<"\nstatus=PASS\n";
        return 0;
    } catch(const std::exception& e){return fail(e.what());}
    catch(...){return fail("unexpected evaluation failure");}
}
} // namespace micius::i4
