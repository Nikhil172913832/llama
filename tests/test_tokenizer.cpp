#include <filesystem>
#include <string>
#include <vector>

#include "doctest.h"
#include "tokenizer.h"

TEST_CASE("testing tokenizer constructor") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "tokenizer.bin";
    CHECK_NOTHROW(llama::Tokenizer(p, 32000));
    CHECK_THROWS_AS(llama::Tokenizer("nonexistent_tokenizer.bin", 32000),
                    std::runtime_error);
    CHECK_THROWS_AS(llama::Tokenizer(p, 31999), std::runtime_error);
    CHECK_THROWS_AS(llama::Tokenizer(p, 32001), std::runtime_error);
}

TEST_CASE("testing special tokens") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "tokenizer.bin";
    llama::Tokenizer t(p, 32000);

    CHECK(t.decode(0) == "<unk>");
    CHECK(t.decode(1) == "\n<s>\n");
    CHECK(t.decode(2) == "\n</s>\n");
    CHECK(t.decode(3) == "<0x00>");
    CHECK(t.decode(258) == "<0xFF>");
}

TEST_CASE("testing encode and decode reference tokens") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "tokenizer.bin";
    llama::Tokenizer t(p, 32000);
    std::vector<int> expected_tokens = {1, 9038, 2501, 263, 931};
    std::vector<int> actual_tokens = t.encode("Once upon a time", true);
    CHECK(actual_tokens == expected_tokens);
    std::string decoded = "";
    int prev_token = -1;
    for (int tok_id : actual_tokens) {
        if (tok_id != 1) {
            decoded += std::string(t.decode(tok_id, prev_token));
        }
        prev_token = tok_id;
    }
    CHECK(decoded == "Once upon a time");
}

TEST_CASE("testing empty string edge cases") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "tokenizer.bin";
    llama::Tokenizer t(p, 32000);
    std::vector<int> with_bos = t.encode("", true);
    CHECK(with_bos.size() == 1);
    CHECK(with_bos[0] == 1);
    std::vector<int> without_bos = t.encode("", false);
    CHECK(without_bos.empty());
}

TEST_CASE("testing round trip arbitrary text") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "tokenizer.bin";
    llama::Tokenizer t(p, 32000);
    std::vector<std::string> test_prompts = {
        "The quick brown fox jumps over the lazy dog.",
        "Hello, World! 12345 @#$%",
        "In a faraway kingdom, there lived a tiny dragon.",
        "    leading and trailing spaces    ",
    };
    for (const auto& prompt : test_prompts) {
        auto tokens = t.encode(prompt, true);
        std::string reconstructed = "";
        int prev = -1;
        for (int tok_id : tokens) {
            if (tok_id != 1) {
                reconstructed += std::string(t.decode(tok_id, prev));
            }
            prev = tok_id;
        }
        CHECK(reconstructed == prompt);
    }
}
