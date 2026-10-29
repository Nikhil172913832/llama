#include <cstddef>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "ops.h"
#include "test_utils.h"
#include <vector>
TEST_CASE("testing rmsnorm") {
    std::vector<float> data =
        read("/home/darklord/Work/Projects/llama/testdata/rmsnorm.bin");
    const float *x = data.data();
    const float *w = data.data() + 6 * 288;
    const float *expected = data.data() + 7 * 288;
    std::size_t size = 6 * 288;
    std::vector<float> actual(size);
    rmsnorm(x, w, actual.data(), 1e-5f, 6, 288);
    float atol = 1e-5;
    float rtol = 1.3e-6;
    CHECK(compare(actual.data(), expected, size, atol, rtol));
}
TEST_CASE("testing matvec") {
    std::vector<float> data =
        read("/home/darklord/Work/Projects/llama/testdata/matvec.bin");
    const float *x = data.data();
    const float *w = data.data() + 288;
    const float *expected = data.data() + 513 * 288;
    std::size_t size = 512;
    std::vector<float> actual(size);
    matvec(x, w, actual.data(), 512, 288);
    float atol = 3e-5;
    float rtol = 1.3e-6;
    CHECK(compare(actual.data(), expected, size, atol, rtol));
}
TEST_CASE("testing softmax") {
    std::vector<float> data =
        read("/home/darklord/Work/Projects/llama/testdata/softmax.bin");
    const float *expected = data.data() + 4 * 288;
    std::size_t size = 4 * 288;
    std::vector<float> actual(data.begin(), data.begin() + size);
    for (int i = 0; i < 4; i++) {
        softmax(actual.data() + i * 288, 288);
    }
    float atol = 1e-5;
    float rtol = 1.3e-6;
    CHECK(compare(actual.data(), expected, size, atol, rtol));
}