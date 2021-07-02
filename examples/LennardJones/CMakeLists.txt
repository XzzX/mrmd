configure_file(positions.txt . COPYONLY)
add_executable(LennardJones
        LennardJones.cpp
        )
target_link_libraries(LennardJones
        PRIVATE mrmd CLI11::CLI11
        )