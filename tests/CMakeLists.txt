add_executable(llama_tests
    test_ops.cpp
    test_utils.cpp
    test_mem.cpp
    test_model.cpp
    test_tokenizer.cpp
)

target_link_libraries(llama_tests
    PRIVATE engine
)

add_test(NAME llama_tests COMMAND llama_tests)

target_compile_definitions(llama_tests
    PRIVATE
        TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)
