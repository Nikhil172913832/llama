cmake_minimum_required(VERSION 4.4)

project(inference_engine
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(ops
    src/ops.cpp
)

target_include_directories(ops
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
