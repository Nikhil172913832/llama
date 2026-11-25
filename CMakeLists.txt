cmake_minimum_required(VERSION 4.5)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(inference_engine
    VERSION 1.0
    LANGUAGES CXX
)

find_package(OpenMP REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(engine
    src/ops.cpp
    src/mem.cpp
    src/model.cpp
    src/tokenizer.cpp
)

target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(engine
    PRIVATE -march=native
)
target_include_directories(engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(CTest)

add_executable(llama_run
  src/main.cpp
)
target_link_libraries(llama_run PRIVATE engine)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
