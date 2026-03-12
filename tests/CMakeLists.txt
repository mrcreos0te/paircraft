add_executable(paircraft_tests
    test_bpe.cpp
)

target_link_libraries(paircraft_tests
    PRIVATE
        paircraft
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(paircraft_tests)
