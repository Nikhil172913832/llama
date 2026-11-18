#include <chrono>
#include <filesystem>
#include <iostream>

#include "doctest.h"
#include "model.h"

TEST_CASE("benchmark stories15M") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "stories15M.bin";
    llama::Engine engine(p);
    int token_count = 0;
    auto on_token = [&](int token_id) {
        token_count++;
        return token_count < 100;
    };
    std::vector<int> prompt = {1};
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.generate(prompt, on_token);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration<double>(t1 - t0).count();
    auto token_per_sec = token_count / time_taken;
    std::cout << "Time Taken :" << time_taken << std::endl;
    std::cout << "Token per sec :" << token_per_sec << std::endl;
}
