add_executable(BubbleLJ
        BubbleLJ.cpp
        )
target_link_libraries(BubbleLJ
        PRIVATE mrmd CLI11::CLI11
        )
