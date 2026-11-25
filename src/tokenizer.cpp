#include <tokenizer.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace {
struct ListNode {
    int32_t prev;
    int32_t next;
    int32_t token_id;
    bool active = true;
};
struct MergeNode {
    int32_t first;
    int32_t second;
    int32_t first_token_id;
    int32_t second_token_id;
    int32_t token_id;
    float score;
    bool operator<(const MergeNode& other) const {
        if (score == other.score) {
            return first > other.first;
        }
        return score < other.score;
    }
};
}  // namespace
namespace llama {
Tokenizer::Tokenizer(const std::filesystem::path& tokenizer_path,
                     std::size_t vocab_size) {
    std::ifstream file(tokenizer_path.c_str(), std::ios::binary);
    if (file) {
        file.read(reinterpret_cast<char*>(&max_token_length), sizeof(uint32_t));
        scores.reserve(vocab_size);
        vocab_offsets.reserve(vocab_size + 1);
        vocab_map.reserve(vocab_size);
        std::size_t vocab_bytes_size =
            std::filesystem::file_size(tokenizer_path) - 4 - (8 * vocab_size);
        vocab_bytes.reserve(vocab_bytes_size);
        std::size_t current_size{};
        vocab_offsets.push_back(0);
        float temp_score{};
        int32_t len{};
        std::size_t write_cursor{};
        while (current_size < vocab_size) {
            if (file.read(reinterpret_cast<char*>(&temp_score),
                          sizeof(float))) {
                if (file.read(reinterpret_cast<char*>(&len), sizeof(len))) {
                    if (len <= 0) {
                        throw std::runtime_error("Invalid size for a token");
                    }
                    if (write_cursor + len > vocab_bytes_size) {
                        throw std::runtime_error(
                            "Token is too large to fit in");
                    }
                    file.read(vocab_bytes.data() + vocab_offsets.back(), len);
                    write_cursor += len;
                    vocab_offsets.push_back(write_cursor);
                    current_size++;
                    scores.push_back(temp_score);
                } else {
                    throw std::runtime_error("Tokenizer file is corrupted");
                }
            } else {
                throw std::runtime_error("Contains fewer tokens than expected");
            }
        }
        char dummy;
        if (file.read(&dummy, 1) || write_cursor != vocab_bytes_size) {
            throw std::runtime_error("Contains more tokens than expected");
        }
        for (std::size_t i = 0; i < vocab_size; i++) {
            std::string_view token(vocab_bytes.data() + vocab_offsets[i],
                                   vocab_offsets[i + 1] - vocab_offsets[i]);
            vocab_map[token] = static_cast<int32_t>(i);
        }
    } else {
        throw std::runtime_error("Failed to open tokenizer file: " +
                                 tokenizer_path.string());
    }
}
std::string_view Tokenizer::decode(const int32_t token_id,
                                   const int32_t prev_token) const {
    std::string_view result =
        std::string_view(vocab_bytes.data() + vocab_offsets[token_id],
                         vocab_offsets[token_id + 1] - vocab_offsets[token_id]);
    if (prev_token == 1 && !result.empty() && result[0] == ' ') {
        result.remove_prefix(1);
    }
    return result;
}
std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                       bool bos) const {
    if (text.size() == 0) {
        if (bos) {
            return {1};
        }
        return std::vector<int>();
    }
    std::vector<int32_t> tokens(text.size() + 1);
    tokens[0] = vocab_map.at(std::string_view(" "));
    for (int32_t i = 0; i < text.size(); i++) {
        std::string_view text_sv = std::string_view(&text[i], 1);
        if (vocab_map.find(text_sv) != vocab_map.end()) {
            tokens[i + 1] = vocab_map.at(std::string_view(&text[i], 1));
        } else {
            tokens[i + 1] = (unsigned char)text[i] + 3;
        }
    }
    std::vector<ListNode> nodes;
    nodes.reserve(tokens.size());
    for (int32_t i = 0; i < tokens.size(); i++) {
        nodes.push_back(ListNode{i - 1, i + 1, tokens[i]});
    }
    std::priority_queue<MergeNode> heap;
    std::vector<char> buffer;
    buffer.reserve(max_token_length);
    for (int32_t i = 0; i < tokens.size() - 1; i++) {
        buffer.clear();
        buffer.insert(buffer.end(),
                      vocab_bytes.data() + vocab_offsets[tokens[i]],
                      vocab_bytes.data() + vocab_offsets[tokens[i] + 1]);
        buffer.insert(buffer.end(),
                      vocab_bytes.data() + vocab_offsets[tokens[i + 1]],
                      vocab_bytes.data() + vocab_offsets[tokens[i + 1] + 1]);
        std::string_view buffer_sv(buffer.data(), buffer.size());
        if (vocab_map.find(buffer_sv) != vocab_map.end()) {
            int32_t token_id = vocab_map.at(buffer_sv);
            heap.push(MergeNode{.first = i,
                                .second = i + 1,
                                .first_token_id = tokens[i],
                                .second_token_id = tokens[i + 1],
                                .token_id = token_id,
                                .score = scores[token_id]});
        }
    }
    while (!heap.empty()) {
        MergeNode candidate = heap.top();
        heap.pop();
        if (nodes[candidate.first].active && nodes[candidate.second].active &&
            nodes[candidate.first].token_id == candidate.first_token_id &&
            nodes[candidate.second].token_id == candidate.second_token_id) {
            nodes[candidate.first].token_id = candidate.token_id;
            nodes[candidate.first].next = nodes[candidate.second].next;
            if (nodes[candidate.second].next != tokens.size()) {
                nodes[nodes[candidate.second].next].prev = candidate.first;
            }
            nodes[candidate.second].active = false;
            if (nodes[candidate.first].prev != -1) {
                buffer.clear();
                buffer.insert(
                    buffer.end(),
                    vocab_bytes.data() +
                        vocab_offsets[nodes[nodes[candidate.first].prev]
                                          .token_id],
                    vocab_bytes.data() +
                        vocab_offsets
                            [nodes[nodes[candidate.first].prev].token_id + 1]);
                buffer.insert(
                    buffer.end(),
                    vocab_bytes.data() +
                        vocab_offsets[nodes[candidate.first].token_id],
                    vocab_bytes.data() +
                        vocab_offsets[nodes[candidate.first].token_id + 1]);
                std::string_view buffer_sv(buffer.data(), buffer.size());
                if (vocab_map.find(buffer_sv) != vocab_map.end()) {
                    int32_t token_id = vocab_map.at(buffer_sv);
                    heap.push(MergeNode{
                        .first = nodes[candidate.first].prev,
                        .second = candidate.first,
                        .first_token_id =
                            nodes[nodes[candidate.first].prev].token_id,
                        .second_token_id = nodes[candidate.first].token_id,
                        .token_id = token_id,
                        .score = scores[token_id]});
                }
            }
            if (nodes[candidate.first].next != tokens.size()) {
                buffer.clear();
                buffer.insert(
                    buffer.end(),
                    vocab_bytes.data() +
                        vocab_offsets[nodes[candidate.first].token_id],
                    vocab_bytes.data() +
                        vocab_offsets[nodes[candidate.first].token_id + 1]);
                buffer.insert(
                    buffer.end(),
                    vocab_bytes.data() +
                        vocab_offsets[nodes[nodes[candidate.first].next]
                                          .token_id],
                    vocab_bytes.data() +
                        vocab_offsets
                            [nodes[nodes[candidate.first].next].token_id + 1]);
                std::string_view buffer_sv =
                    std::string_view(buffer.data(), buffer.size());
                if (vocab_map.find(buffer_sv) != vocab_map.end()) {
                    int32_t token_id = vocab_map.at(buffer_sv);
                    heap.push(MergeNode{
                        .first = candidate.first,
                        .second = nodes[candidate.first].next,
                        .first_token_id = nodes[candidate.first].token_id,
                        .second_token_id =
                            nodes[nodes[candidate.first].next].token_id,
                        .token_id = token_id,
                        .score = scores[token_id]});
                }
            }
        }
    }
    std::vector<int> result;
    if (bos) {
        result.push_back(1);
    }
    for (const auto& node : nodes) {
        if (node.active) {
            result.push_back(node.token_id);
        }
    }
    return result;
}
}  // namespace llama
