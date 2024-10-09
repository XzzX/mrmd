# Copyright 2024 Sebastian Eibl
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(mrmd_add_test)
    if (BUILD_TESTING AND MRMD_ENABLE_TESTING)
        set(options)
        set(oneValueArgs NAME PROCESSES)
        set(multiValueArgs FILES LABELS)
        cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        if (NOT ARG_NAME)
            message(FATAL_ERROR "No name given for test!")
        endif ()

        if (NOT ARG_NAME)
            message(FATAL_ERROR "No files given for test ${ARG_NAME}!")
        endif ()

        if (NOT ARG_PROCESSES)
            set(num_processes 1)
        else ()
            set(num_processes ${ARG_PROCESSES})
        endif ()

        add_executable(${ARG_NAME}
                ${ARG_FILES}
                )
        target_link_libraries(${ARG_NAME}
                PRIVATE mrmd_test_main
                )
        if (MRMD_ENABLE_MPI)
            add_test(NAME ${ARG_NAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_processes} --oversubscribe --allow-run-as-root ${MPIEXEC_PREFLAGS} ./${ARG_NAME})
        else ()
            add_test(NAME ${ARG_NAME} COMMAND ./${ARG_NAME})
        endif()

        set_tests_properties(${ARG_NAME} PROPERTIES ENVIRONMENT "OMP_PROC_BIND=spread;OMP_PLACES=threads")
        if (NOT ARG_LABELS)
            set_tests_properties(${ARG_NAME} PROPERTIES LABELS "${ARG_LABELS}")
        endif ()
        if (MRMD_ENABLE_MPI)
            set_tests_properties(${ARG_NAME} PROPERTIES PROCESSORS ${num_processes})
            set_tests_properties(${ARG_NAME} PROPERTIES PROCESSOR_AFFINITY ON)
        endif()
    endif ()
endfunction()
