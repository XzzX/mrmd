add_executable(ConstraintsIntegrationTest
        Constraints.cpp
        )
target_link_libraries(ConstraintsIntegrationTest
        PRIVATE mrmd gtest
        )

include(GoogleTest)
gtest_discover_tests(ConstraintsIntegrationTest
        DISCOVERY_MODE PRE_TEST)
