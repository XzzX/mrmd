file(GLOB_RECURSE MRMD_SOURCES *.hpp *.cpp)
file(GLOB_RECURSE MRMD_TESTS *test*)

list(REMOVE_ITEM MRMD_SOURCES ${MRMD_TESTS})

set(MRMD_VECTOR_LENGTH 1 CACHE STRING "vector length of AoSoA data structure")
configure_file(cmake.hpp.in cmake.hpp @ONLY)
configure_file(version.hpp.in version.hpp @ONLY)

add_library(mrmd ${MRMD_SOURCES})
target_link_libraries(mrmd
        PUBLIC Kokkos::kokkos
        PUBLIC Cabana::cabanacore
        PUBLIC MPI::MPI_C MPI::MPI_CXX
        PRIVATE fmt::fmt
        )
if (MRMD_HDF5_AVAILABLE)
    target_link_libraries(mrmd PUBLIC hdf5::hdf5 hdf5::hdf5_hl)
endif ()
target_include_directories(mrmd
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

if (MRMD_WERROR)
    target_compile_options(mrmd PRIVATE "-Werror")
endif ()
# compiler specific configurations
if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    target_compile_options(mrmd PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if (MRMD_LOCAL_ARCHITECTURE)
        target_compile_options(mrmd PRIVATE "-march=native")
    endif ()
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    target_compile_options(mrmd PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if (MRMD_ENABLE_COVERAGE)
        target_compile_options(mrmd PUBLIC -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(mrmd PUBLIC -fprofile-instr-generate -fcoverage-mapping)
    endif ()
    if (MRMD_LOCAL_ARCHITECTURE)
        target_compile_options(mrmd PRIVATE "-march=native")
    endif ()
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    #https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/floating-point-operations/understanding-floating-point-operations/floating-point-optimizations.html
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 2021.4.0)
        target_compile_options(mrmd PRIVATE "-fp-model=strict")
    endif ()
    if (MRMD_LOCAL_ARCHITECTURE)
        target_compile_options(mrmd PRIVATE "-xHost")
    endif ()
    if (MRMD_VEC_REPORT)
        target_compile_options(mrmd PRIVATE -qopt-report-phase=vec,loop -qopt-report=5)
        #target_compile_options(mrmd PRIVATE -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
    endif ()
endif ()

install(TARGETS mrmd
        EXPORT mrmd-export
        )
install(EXPORT mrmd-export
        NAMESPACE mrmd::
        DESTINATION lib/cmake/mrmd
        FILE mrmdTargets.cmake
        )
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        TYPE INCLUDE
        FILES_MATCHING PATTERN "*.hpp")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mrmdConfig.cmake
        DESTINATION lib/cmake/mrmd)

if (BUILD_TESTING AND MRMD_ENABLE_TESTING)
    add_library(mrmd_test_main gtest_main.test.cpp)
    target_link_libraries(mrmd_test_main
            PUBLIC mrmd
            PUBLIC gtest)
endif ()

add_subdirectory(action)
add_subdirectory(analysis)
add_subdirectory(communication)
add_subdirectory(data)
add_subdirectory(io)
add_subdirectory(util)
add_subdirectory(weighting_function)
