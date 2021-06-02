file(GLOB_RECURSE ECAB_SOURCES *.hpp *.cpp)
file(GLOB_RECURSE ECAB_TESTS *test*)

list(REMOVE_ITEM ECAB_SOURCES ${ECAB_TESTS})

add_library(ecab ${ECAB_SOURCES})
target_link_libraries(ecab PUBLIC Kokkos::kokkos Cabana::cabanacore)
target_include_directories(ecab
     PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)

if (ECAB_WERROR)
    target_compile_options(ecab PRIVATE "-Werror")
endif ()
# compiler specific configurations
if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    target_compile_options(ecab PRIVATE -Wall -Wextra -Wpedantic)
    if (ECAB_LOCAL_ARCHITECTURE)
        target_compile_options(ecab PRIVATE "-march=native")
    endif ()
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    target_compile_options(ecab PRIVATE -Wall -Wextra -Wpedantic)
    if (ECAB_LOCAL_ARCHITECTURE)
        target_compile_options(ecab PRIVATE "-march=native")
    endif ()
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    if (ECAB_LOCAL_ARCHITECTURE)
        target_compile_options(ecab PRIVATE "-xHost")
    endif ()
    if (ECAB_VEC_REPORT)
        target_compile_options(ecab PRIVATE -qopt-report-phase=vec,loop -qopt-report=5)
        #target_compile_options(ecab PRIVATE -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
    endif ()
endif ()

add_executable(ecab_tests
        ${ECAB_TESTS}
        )
target_link_libraries(ecab_tests
        PRIVATE ecab gtest
        )

include(GoogleTest)
gtest_discover_tests(ecab_tests)
