configure_file(positions.txt . COPYONLY)
add_executable(LennardJonesIntegrationTest
        LennardJones.cpp
        )
target_link_libraries(LennardJonesIntegrationTest
        PRIVATE mrmd gtest
        )

include(GoogleTest)
gtest_discover_tests(LennardJonesIntegrationTest
        DISCOVERY_MODE PRE_TEST)