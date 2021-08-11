configure_file(positions.txt . COPYONLY)
configure_file(push_performance.py . COPYONLY)
add_executable(LennardJones
        LennardJones.cpp
        )
target_link_libraries(LennardJones
        PRIVATE mrmd CLI11::CLI11
        )