add_executable(test_ops
    test_ops.cpp
    test_utils.cpp
)

target_link_libraries(test_ops
    PRIVATE ops
)

add_test(NAME ops_tests COMMAND test_ops)

target_compile_definitions(test_ops
    PRIVATE
        TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)
