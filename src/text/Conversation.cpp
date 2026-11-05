#include "Conversation.h"
#include <algorithm>
#include <stdexcept>

namespace micius::text {
Conversation::Conversation(MiciusContext* ctx, size_t max_context, const Tokenizer& tok)
    : engine(ctx), capacity(max_context), tokenizer(tok)
{
    if (!engine || !tok.SupportsChat())
        throw std::invalid_argument("Chat needs a loaded model with a supported GGUF chat template");
    Reset();
}
void Conversation::Reset()
{
    if (!Micius_ResetEngineV1(engine)) throw std::runtime_error(Micius_GetLastError());
    history.clear(); dirty=false;
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
    uint32_t next=Decode(staged.back());
    size_t emitted=0;
    while (emitted<budget && !tokenizer.IsStop(next)) {
        if (cancel.load(std::memory_order_relaxed)) return {Finish::Interrupted,emitted};
        emit(tokenizer.Piece(next));
        staged.push_back(next); ++emitted;
        next=Decode(next);
    }
    if (cancel.load(std::memory_order_relaxed)) return {Finish::Interrupted,emitted};
    for (uint32_t id:close) { Prefill(id); staged.push_back(id); }
    if (Micius_GetSequenceLengthV1(engine)!=history.size()+staged.size())
        throw std::runtime_error("Chat token history/KV length mismatch");
    history.insert(history.end(),staged.begin(),staged.end());
    dirty=false;
    return {tokenizer.IsStop(next) ? Finish::End :
            (space<limit ? Finish::ContextLimit : Finish::ReplyLimit),emitted};
}
}
