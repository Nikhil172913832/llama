#pragma once
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
namespace llama {
class Tokenizer {
   public:
    explicit Tokenizer(const std::filesystem::path& tokenizer_path,
                       const std::size_t vocab_size = 32000);
    std::vector<int> encode(const std::string& text, bool bos = true) const;
    std::string_view decode(const int token_id,
                            const int prev_token = -1) const;

   private:
    std::unordered_map<std::string_view, int32_t> vocab_map;
    std::vector<char> vocab_bytes;
    std::vector<std::size_t> vocab_offsets;
    std::vector<float> scores;
    uint32_t max_token_length{};
};
}  // namespace llama
