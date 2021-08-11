file(GLOB_RECURSE MRMD_SOURCES *.hpp *.cpp)
file(GLOB_RECURSE MRMD_TESTS *test*)

list(REMOVE_ITEM MRMD_SOURCES ${MRMD_TESTS})

add_library(mrmd ${MRMD_SOURCES})
target_link_libraries(mrmd PUBLIC Kokkos::kokkos Cabana::cabanacore)
target_include_directories(mrmd
        PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)

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
    if (MRMD_LOCAL_ARCHITECTURE)
        target_compile_options(mrmd PRIVATE "-march=native")
    endif ()
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    if (MRMD_LOCAL_ARCHITECTURE)
        target_compile_options(mrmd PRIVATE "-xHost")
    endif ()
    if (MRMD_VEC_REPORT)
        target_compile_options(mrmd PRIVATE -qopt-report-phase=vec,loop -qopt-report=5)
        #target_compile_options(mrmd PRIVATE -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
    endif ()
endif ()

add_executable(mrmd_tests
        ${MRMD_TESTS}
        )
target_link_libraries(mrmd_tests
        PRIVATE mrmd gtest
        )

include(GoogleTest)
gtest_discover_tests(mrmd_tests
        DISCOVERY_MODE PRE_TEST)
