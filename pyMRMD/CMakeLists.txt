set(PYBIND11_FINDPYTHON ON)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.9.0
)
FetchContent_MakeAvailable(pybind11)

include(CMakePrintHelpers)
find_package(Python3 COMPONENTS Interpreter Development)
cmake_print_variables(Python3_SITEARCH)

pybind11_add_module(pyMRMD
        pyMRMD.cpp
        action.cpp
        analysis.cpp
        cabana.cpp
        communication.cpp
        data.cpp
        io.cpp
        util.cpp)
target_link_libraries(pyMRMD PRIVATE mrmd)