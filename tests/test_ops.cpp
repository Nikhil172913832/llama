#include <cstddef>
#include <filesystem>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <vector>

#include "doctest.h"
#include "ops.h"
#include "test_utils.h"
using namespace llama;
TEST_CASE("testing rmsnorm") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "rmsnorm.bin";
    std::vector<float> data = read(p);
    const float* x = data.data();
    const float* w = data.data() + 6 * 288;
    const float* expected = data.data() + 7 * 288;
    std::size_t size = 6 * 288;
    std::vector<float> actual(size);
    rmsnorm(x, w, actual.data(), 1e-5f, 6, 288);
    const float atol = 1e-5;
    const float rtol = 1.3e-6;
    CHECK(compare(actual.data(), expected, size, atol, rtol));
}
TEST_CASE("testing matvec") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "matvec.bin";
    std::vector<float> data = read(p);
    const float* x = data.data();
    const float* w = data.data() + 288;
    const float* expected = data.data() + 513 * 288;
    std::size_t size = 512;
    std::vector<float> actual(size);
    matvec(x, w, actual.data(), 512, 288);
    const float atol = 3e-5;
    const float rtol = 1.3e-6;
    CHECK(compare(actual.data(), expected, size, atol, rtol));
}
TEST_CASE("testing softmax") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "softmax.bin";
    std::vector<float> data = read(p);
    const float* expected = data.data() + 4 * 288;
    std::size_t size = 4 * 288;
    std::vector<float> actual(data.begin(), data.begin() + size);
    for (int i = 0; i < 4; i++) {
        softmax(actual.data() + i * 288, 288);
    }
    const float atol = 1e-5;
    const float rtol = 1.3e-6;
    CHECK(compare(actual.data(), expected, size, atol, rtol));
}
TEST_CASE("testing rope") {
    std::filesystem::path p = std::filesystem::path(TEST_DATA_DIR) / "rope.bin";
    std::vector<float> data = read(p);
    float* xq = data.data();
    float* xk = data.data() + 16 * 4 * 8;
    const float* xq_r = data.data() + 16 * 4 * 8 + 16 * 2 * 8;
    const float* xk_r = data.data() + 2 * 16 * 4 * 8 + 16 * 2 * 8;
    const float* cp = data.data() + 2 * 16 * 4 * 8 + 2 * 16 * 2 * 8;
    const float* sp = data.data() + 2 * 16 * 4 * 8 + 2 * 16 * 2 * 8 + 16 * 4;
    std::vector<float> c(16 * 4);
    std::vector<float> s(16 * 4);
    const float atol = 1e-5;
    const float rtol = 1.3e-6;
    precompute_freq_cis(8, 16, c.data(), s.data());
    CHECK(compare(c.data(), cp, 16 * 4, atol, rtol));
    CHECK(compare(s.data(), sp, 16 * 4, atol, rtol));
    rope(xq, xk, c.data(), s.data(), 16, 4, 2, 8, 0);
    CHECK(compare(xq, xq_r, 16 * 4 * 8, atol, rtol));
    CHECK(compare(xk, xk_r, 16 * 2 * 8, atol, rtol));
}
TEST_CASE("testing swiglu") {
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "swiglu.bin";
    std::vector<float> data = read(p);
    float* x = data.data();
    float* up = data.data() + 5 * 4;
    float* x_s = data.data() + 2 * 5 * 4;
    swiglu(up, x, 5 * 4);
    const float atol = 1e-5;
    const float rtol = 1.3e-6;
    CHECK(compare(x, x_s, 5 * 4, atol, rtol));
}
TEST_CASE("testing attention"){
    std::filesystem::path p =
        std::filesystem::path(TEST_DATA_DIR) / "attention.bin";
    std::vector<float> data = read(p);
    float* k_cache = data.data();
    float* v_cache = data.data() + 2*16*8;
    float* q = data.data() + 2*2*16*8;
    float* outp = data.data() + 2*2*16*8 + 4*8;
    std::vector<float> out(4*8);
    std::vector<float> scores(6);
    const float atol = 1e-5;
    const float rtol = 1.3e-6;
    attention(k_cache, v_cache, q, scores.data(),out.data(), 5, 4, 2, 8, 16);
    CHECK(compare(out.data(), outp, 4*8, atol, rtol));
}