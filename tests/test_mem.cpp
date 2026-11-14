#include <stdexcept>
#include "doctest.h"
#include "mem.h"
using namespace llama;
TEST_CASE("testing out of memory runtime expection"){
    llama::MemoryArena arena(10);
    CHECK_THROWS_AS(arena.alloc(11), std::runtime_error);
}
TEST_CASE("testing alloc"){
    llama::MemoryArena arena(10);
    
}
