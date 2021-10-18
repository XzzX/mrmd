add_executable(NPT
        NPT.cpp
        )
target_link_libraries(NPT
        PRIVATE mrmd gtest
        )

include(GoogleTest)
gtest_discover_tests(NPT
        DISCOVERY_MODE PRE_TEST)
