configure_file(positions.txt . COPYONLY)
add_executable(LennardJonesIntegrationTest
        LennardJones.cpp
        )
target_link_libraries(LennardJonesIntegrationTest
        PRIVATE ecab gtest
        )

include(GoogleTest)
gtest_discover_tests(LennardJonesIntegrationTest)