#include <chrono>
#include <filesystem>
#include <iostream>

#include "doctest.h"
#include "model.h"
#include "tokenizer.h"

TEST_CASE("benchmark stories15M") {
    std::filesystem::path model_path =
        std::filesystem::path(TEST_DATA_DIR) / "stories15M.bin";
    std::filesystem::path tok_path =
        std::filesystem::path(TEST_DATA_DIR) / "tokenizer.bin";

    llama::Engine engine(model_path);
    llama::Tokenizer tokenizer(tok_path);

    std::string prompt_str = "One day, a little dog named Spot";
    std::vector<int> prompt = tokenizer.encode(prompt_str, true);

    std::cout << "\n================== Model Output ==================\n";
    std::cout << prompt_str;

    int prev_token = prompt.back();
    int generated_count = 0;

    auto on_token = [&](int token_id) {
        if (token_id == 2) {
            return false;
        }
        std::string_view piece = tokenizer.decode(token_id, prev_token);
        std::cout << piece << std::flush;
        prev_token = token_id;
        generated_count++;
        return generated_count < 256;
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    engine.generate(prompt, on_token);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    auto tok_per_sec = generated_count / elapsed;

    std::cout << "\n==================================================\n";
    std::cout << "Generated Tokens : " << generated_count << "\n";
    std::cout << "Time Taken       : " << elapsed << " s\n";
    std::cout << "Throughput       : " << tok_per_sec << " tok/s\n";
    std::cout << "==================================================\n";
}
