add_executable(Argon
    Argon.cpp
        )
    target_link_libraries(Argon
        PRIVATE mrmd CLI11::CLI11
        )
    add_test(NAME Example.Argon COMMAND ./Argon --nsteps 10)
