#pragma once
#include "Tokenizer.h"
#include "micius_runtime.h"
#include <atomic>
#include <functional>

namespace micius::text {
enum class Finish { End, ReplyLimit, ContextLimit, Interrupted };
struct Answer { Finish reason; size_t tokens; };

class Conversation {
public:
    Conversation(MiciusContext* engine, size_t capacity, const Tokenizer& tokenizer);
    void Reset();
    Answer Reply(std::string_view user, size_t limit, const std::atomic<bool>& cancel,
                 const std::function<void(std::string_view)>& emit);
private:
    MiciusContext* engine;
    size_t capacity;
    const Tokenizer& tokenizer;
    // A completed turn is the rollback point for a failed/cancelled turn.
    std::vector<uint32_t> history;
    bool dirty = false;
    void Prefill(uint32_t id) const;
    uint32_t Decode(uint32_t id) const;
};
}
