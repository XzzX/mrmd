add_executable(SPC
        SPC.cpp
        )
target_link_libraries(SPC
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
