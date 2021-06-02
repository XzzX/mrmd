add_executable(vectorization
        vec.cpp
        )
target_link_libraries(vectorization
        PRIVATE ecab
        )
