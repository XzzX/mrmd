add_executable(LangevinThermostat
    LangevinThermostat.cpp
        )
    target_link_libraries(LangevinThermostat
        PRIVATE mrmd CLI11::CLI11
        )
    add_test(NAME IntegrationTest.LangevinThermostat COMMAND ./LangevinThermostat )
