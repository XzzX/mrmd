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
        add_test(NAME ${ARG_NAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_processes} --oversubscribe --allow-run-as-root ${MPIEXEC_PREFLAGS} ./${ARG_NAME})

        set_tests_properties(${ARG_NAME} PROPERTIES ENVIRONMENT "OMP_PROC_BIND=spread;OMP_PLACES=threads")
        if (NOT ARG_LABELS)
            set_tests_properties(${ARG_NAME} PROPERTIES LABELS "${ARG_LABELS}")
        endif ()
        set_tests_properties(${ARG_NAME} PROPERTIES PROCESSORS ${num_processes})
        set_tests_properties(${ARG_NAME} PROPERTIES PROCESSOR_AFFINITY ON)
    endif ()
endfunction()
