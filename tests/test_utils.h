#include <cstddef>
#include <string>
#include <vector>
bool compare(const float *actual, const float *expected, const std::size_t size,
             const float atol, const float rtol);
std::vector<float> read(const std::string &filepath);
