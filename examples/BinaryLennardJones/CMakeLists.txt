configure_file(input.yaml . COPYONLY)
add_executable(BinaryLennardJones
        BinaryLennardJones.cpp
        initialization.cpp
        NPT.cpp
        SPARTIAN.cpp
        )
target_link_libraries(BinaryLennardJones
        PRIVATE mrmd CLI11::CLI11 fmt::fmt yaml-cpp::yaml-cpp
        )
#add_test(NAME Example.BinaryLennardJones COMMAND ./BinaryLennardJones --nsteps 10)
