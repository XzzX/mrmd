add_executable(NVT
        NVT.cpp
        )
target_link_libraries(NVT
        PRIVATE mrmd gtest
        )

include(GoogleTest)
gtest_discover_tests(NVT
        DISCOVERY_MODE PRE_TEST)
