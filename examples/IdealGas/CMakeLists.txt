add_executable(IdealGas
        IdealGas.cpp
        )
target_link_libraries(IdealGas
        PRIVATE mrmd CLI11::CLI11
        )
