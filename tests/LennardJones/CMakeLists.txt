configure_file(positions.txt . COPYONLY)
add_executable(LennardJonesIntegrationTest
        LennardJones.cpp
        )
target_link_libraries(LennardJonesIntegrationTest
        PRIVATE mrmd gtest
        )