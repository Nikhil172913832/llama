#pragma once
#include <cstddef>
#include <vector>
namespace llama {
class MemoryArena {
   public:
    explicit MemoryArena(size_t n);
    float* alloc(size_t n);

   private:
    std::vector<float> storage;
    std::size_t cursor;
};
}  // namespace llama