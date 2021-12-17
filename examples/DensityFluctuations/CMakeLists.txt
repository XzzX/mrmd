add_executable(DensityFluctuations
    DensityFluctuations.cpp
        )
    target_link_libraries(DensityFluctuations
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
    add_test(NAME Example.DensityFluctuations COMMAND ./DensityFluctuations --nsteps 10)
