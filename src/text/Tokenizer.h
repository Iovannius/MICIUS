#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class GGUFParser;

namespace micius::text {
class Tokenizer {
public:
    explicit Tokenizer(const GGUFParser& reader);
    std::vector<uint32_t> Encode(std::string_view utf8) const;
    std::string_view Piece(uint32_t id) const;
    bool IsStop(uint32_t id) const { return id == end || id == end_of_text; }
    bool SupportsChat() const { return !system_message.empty(); }
    bool CanGenerate(uint32_t id) const { return IsStop(id) || (id<pieces.size() && !pieces[id].empty()); }
    size_t VocabularySize() const { return pieces.size(); }
    std::vector<uint32_t> Prompt(bool first, std::string_view user) const;
    std::vector<uint32_t> EndTurn() const;

private:
    struct Merge { uint32_t rank, result; };
    std::array<uint32_t,256> bytes{};
    std::vector<std::string> pieces;
    std::vector<uint32_t> user_defined;
    std::unordered_map<uint64_t,Merge> merges;
    uint32_t start{}, end{}, end_of_text{};
    bool qwen{};
    std::string_view system_message;
    void MergePiece(std::string_view piece, std::vector<uint32_t>& out) const;
};
}
