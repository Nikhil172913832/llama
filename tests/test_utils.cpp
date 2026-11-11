#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
bool compare(const float* actual, const float* expected, const std::size_t size,
             const float atol, const float rtol) {
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0) {
        std::cerr << "Tolerance value cannot be infinite or negative"
                  << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < size; i++) {
        float tolerance = atol + rtol * std::fabs(expected[i]);
        float error = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            !std::isfinite(error) || error > tolerance) {
            std::ostringstream message;
            message << "Mismatch at index " << i << ": actual=" << actual[i]
                    << ", expected=" << expected[i] << ", error=" << error;
            std::cerr << message.str() << std::endl;
            return false;
        }
    }
    return true;
}

std::vector<float> read(const std::string& filepath) {
    std::ifstream file(filepath.c_str(), std::ios::binary | std::ios::ate);
    if (file) {
        std::fstream::pos_type size = file.tellg();
        size = size / sizeof(float);
        std::vector<float> result(size);
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(result.data()),
                  result.size() * sizeof(float));
        return result;
    } else {
        std::perror(filepath.c_str());
        return std::vector<float>();
    }
}
