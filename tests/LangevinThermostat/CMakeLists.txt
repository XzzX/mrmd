add_executable(LangevinThermostatIntegrationTest
        LangevinThermostat.cpp
        )
target_link_libraries(LangevinThermostatIntegrationTest
        PRIVATE mrmd gtest
        )
add_test(NAME IntegrationTest.LangevinThermostat COMMAND ./LangevinThermostat)

include(GoogleTest)
gtest_discover_tests(LangevinThermostatIntegrationTest
        DISCOVERY_MODE PRE_TEST)
