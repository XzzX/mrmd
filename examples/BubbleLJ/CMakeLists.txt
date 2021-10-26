configure_file(LJ_spartian_3.lammpstrj . COPYONLY)
add_executable(BubbleLJ
        BubbleLJ.cpp
        )
target_link_libraries(BubbleLJ
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
add_test(NAME Example.BubbleLJ COMMAND ./BubbleLJ --nsteps 10 --output 5)
