mrmd_add_test(NAME test.communication.AccumulateForce
        FILES AccumulateForce.test.cpp)

mrmd_add_test(NAME test.communication.GhostExchange
        FILES GhostExchange.test.cpp)

mrmd_add_test(NAME test.communication.MultiResPeriodicGhostExchange
        FILES MultiResPeriodicGhostExchange.test.cpp)

mrmd_add_test(NAME test.communication.MultiResRealAtomsExchange
        FILES MultiResRealAtomsExchange.test.cpp)

mrmd_add_test(NAME test.communication.PeriodicMapping
        FILES PeriodicMapping.test.cpp)

mrmd_add_test(NAME test.communication.UpdateGhostAtoms
        FILES UpdateGhostAtoms.test.cpp)
