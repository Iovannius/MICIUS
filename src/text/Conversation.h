#pragma once
#include "Tokenizer.h"
#include "micius_runtime.h"
#include "micius_sampling.h"
#include <atomic>
#include <functional>

namespace micius::text {
enum class Finish { End, ReplyLimit, ContextLimit, Interrupted };
struct Answer { Finish reason; size_t tokens; };

class Conversation {
public:
    Conversation(MiciusContext* engine, size_t capacity, const Tokenizer& tokenizer, bool reset=true);
    void Reset();
    void EnsureClean();
    void Configure(const MiciusSamplingOptionsV1& options, uint64_t seed);
    const std::vector<uint32_t>& Tokens() const { return history; }
    uint64_t RandomState() const { return rng; }
    void AdoptRestoredHistory(std::vector<uint32_t>& tokens, uint64_t state) noexcept;
    Answer Reply(std::string_view user, size_t limit, const std::atomic<bool>& cancel,
                 const std::function<void(std::string_view)>& emit);
private:
    MiciusContext* engine;
    size_t capacity;
    const Tokenizer& tokenizer;
    // A completed turn is the rollback point for a failed/cancelled turn.
    std::vector<uint32_t> history;
    bool dirty = false;
    MiciusSamplingOptionsV1 sampling{};
    uint64_t rng=0, initial_seed=0;
    std::vector<float> logits;
    std::vector<uint8_t> allowed;
    void Prefill(uint32_t id) const;
    uint32_t Decode(uint32_t id) const;
    uint32_t Select(uint32_t greedy, const std::vector<uint32_t>& consumed, uint64_t& random);
};
}
