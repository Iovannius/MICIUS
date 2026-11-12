#include "Conversation.h"
#include <algorithm>
#include <stdexcept>

namespace micius::text {
Conversation::Conversation(MiciusContext* ctx, size_t max_context, const Tokenizer& tok,bool reset)
    : engine(ctx), capacity(max_context), tokenizer(tok)
{
    if (!engine || !tok.SupportsChat())
        throw std::invalid_argument("Chat needs a loaded model with a supported GGUF chat template");
    Micius_InitSamplingV1(&sampling,sizeof(sampling));
    if(reset)Reset();
}
void Conversation::Reset()
{
    if (!Micius_ResetEngineV1(engine)) throw std::runtime_error(Micius_GetLastError());
    history.clear(); dirty=false; rng=initial_seed;
}
void Conversation::Configure(const MiciusSamplingOptionsV1& value,uint64_t seed)
{
    float f=1;uint64_t random=seed;uint32_t out=0;
    if(!Micius_SampleTokenV1(&value,&f,1,nullptr,nullptr,0,&random,&out))
        throw std::invalid_argument("Invalid sampling parameters");
    if(seed!=initial_seed)rng=seed;
    sampling=value;initial_seed=seed;
}
void Conversation::EnsureClean()
{
    if(!dirty)return;
    if(!Micius_ResetEngineV1(engine))throw std::runtime_error(Micius_GetLastError());
    for(uint32_t id:history)Prefill(id);
    dirty=false;
}
void Conversation::AdoptRestoredHistory(std::vector<uint32_t>& tokens,uint64_t state) noexcept
{
    history.swap(tokens);rng=state;dirty=false;
}
uint32_t Conversation::Select(uint32_t greedy,const std::vector<uint32_t>& consumed,uint64_t& random)
{
    if(sampling.temperature==0 && (sampling.repeat_penalty==1 || !sampling.repeat_last_n))return greedy;
    logits.resize(tokenizer.VocabularySize());
    if(allowed.empty()) {
        allowed.resize(logits.size());
        for(size_t i=0;i<allowed.size();++i)allowed[i]=tokenizer.CanGenerate(static_cast<uint32_t>(i));
    }
    if(!Micius_CopyLastLogitsV1(engine,logits.data(),logits.size()) ||
       !Micius_SampleTokenV1(&sampling,logits.data(),logits.size(),allowed.data(),consumed.data(),consumed.size(),&random,&greedy))
        throw std::runtime_error("Cannot sample the current logits");
    return greedy;
}
void Conversation::Prefill(uint32_t id) const
{
    if (Micius_PrefillTokenV1(engine,id)!=MICIUS_STATE_TOKEN_ACCEPTED)
        throw std::runtime_error(Micius_GetLastError());
}
uint32_t Conversation::Decode(uint32_t id) const
{
    uint32_t next=0;
    if (Micius_DecodeOneV1(engine,id,&next)!=MICIUS_STATE_YIELD_TOKEN)
        throw std::runtime_error(Micius_GetLastError());
    return next;
}

Answer Conversation::Reply(std::string_view user, size_t limit, const std::atomic<bool>& cancel,
                           const std::function<void(std::string_view)>& emit)
{
    if (limit==0 || user.empty()) throw std::invalid_argument("Expected a nonempty message and positive reply limit");
    auto staged=tokenizer.Prompt(history.empty(),user);
    const auto close=tokenizer.EndTurn();
    const auto free=capacity-history.size();
    if (staged.size()>=free || close.size()>=free-staged.size())
        throw std::length_error("Insufficient context for this turn; use :reset or a shorter message");
    const size_t space=free-staged.size()-close.size();
    const size_t budget=std::min(space,limit);
    auto consumed=history;
    consumed.insert(consumed.end(),staged.begin(),staged.end());
    uint64_t random=rng;

    if (dirty) {
        if (!Micius_ResetEngineV1(engine)) throw std::runtime_error(Micius_GetLastError());
        for (uint32_t id:history) {
            if (cancel.load(std::memory_order_relaxed)) return {Finish::Interrupted,0};
            Prefill(id);
        }
    }
    dirty=true;
    for (size_t i=0; i+1<staged.size(); ++i) {
        if (cancel.load(std::memory_order_relaxed)) return {Finish::Interrupted,0};
        Prefill(staged[i]);
    }
    uint32_t next=Select(Decode(staged.back()),consumed,random);
    size_t emitted=0;
    while (emitted<budget && !tokenizer.IsStop(next)) {
        if (cancel.load(std::memory_order_relaxed)) return {Finish::Interrupted,emitted};
        emit(tokenizer.Piece(next));
        staged.push_back(next); consumed.push_back(next); ++emitted;
        next=Select(Decode(next),consumed,random);
    }
    if (cancel.load(std::memory_order_relaxed)) return {Finish::Interrupted,emitted};
    for (uint32_t id:close) { Prefill(id); staged.push_back(id); }
    if (Micius_GetSequenceLengthV1(engine)!=history.size()+staged.size())
        throw std::runtime_error("Chat token history/KV length mismatch");
    history.insert(history.end(),staged.begin(),staged.end());
    dirty=false; rng=random;
    return {tokenizer.IsStop(next) ? Finish::End :
            (space<limit ? Finish::ContextLimit : Finish::ReplyLimit),emitted};
}
}
