configure_file(LJ_spartian_3.lammpstrj . COPYONLY)
add_executable(SpartianLJ
        SpartianLJ.cpp
        )
target_link_libraries(SpartianLJ
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
add_test(NAME Example.SpartianLJ COMMAND ./SpartianLJ --nsteps 10 --output 5)
