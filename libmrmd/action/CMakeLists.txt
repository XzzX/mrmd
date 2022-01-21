mrmd_add_test(NAME test.action.BerendsenBarostat
        FILES BerendsenBarostat.test.cpp)

mrmd_add_test(NAME test.action.BerendsenThermostat
        FILES BerendsenThermostat.test.cpp)

mrmd_add_test(NAME test.action.ContributeMoleculeForceToAtoms
        FILES ContributeMoleculeForceToAtoms.test.cpp)

mrmd_add_test(NAME test.action.CoulombDSF
        FILES CoulombDSF.test.cpp)

mrmd_add_test(NAME test.action.LangevinThermostat
        FILES LangevinThermostat.test.cpp)

mrmd_add_test(NAME test.action.LennardJones
        FILES LennardJones.test.cpp)

mrmd_add_test(NAME test.action.LimitAcceleration
        FILES LimitAcceleration.test.cpp)

mrmd_add_test(NAME test.action.LimitVelocity
        FILES LimitVelocity.test.cpp)

mrmd_add_test(NAME test.action.LJ_IdealGas
        FILES LJ_IdealGas.test.cpp)

mrmd_add_test(NAME test.action.Shake
        FILES Shake.test.cpp)

mrmd_add_test(NAME test.action.UpdateMolecules
        FILES UpdateMolecules.test.cpp)

mrmd_add_test(NAME test.action.VelocityVerlet
        FILES VelocityVerlet.test.cpp)