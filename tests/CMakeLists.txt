add_executable(tests
    test_ops.cpp
    test_utils.cpp
    test_mem.cpp
    test_model.cpp
)

target_link_libraries(tests
    PRIVATE engine
)

add_test(NAME ops_tests COMMAND tests)

target_compile_definitions(tests
    PRIVATE
        TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)
