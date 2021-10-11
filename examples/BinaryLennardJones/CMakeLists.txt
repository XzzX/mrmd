add_executable(BinaryLennardJones
        BinaryLennardJones.cpp
        )
target_link_libraries(BinaryLennardJones
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
add_test(NAME Example.BinaryLennardJones COMMAND ./BinaryLennardJones --nsteps 10)
