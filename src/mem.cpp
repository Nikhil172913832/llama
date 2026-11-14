#include "mem.h"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace llama {
MemoryArena::MemoryArena(size_t n) {
    storage.resize(n);
    cursor = 0;
}
float* MemoryArena::alloc(size_t n) {
    if(cursor + n <= storage.size()){
        throw std::runtime_error("Arena out of memory");
    }
    float* result = storage.data() + cursor;
    cursor += n;
    return result;
}
}  // namespace llama