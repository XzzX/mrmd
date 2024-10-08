// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "initialization.hpp"

#include <Kokkos_Core.hpp>

#include "mpi_wrapper.hpp"

namespace mrmd
{
void initialize(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
}

void initialize()
{
    int argc = 0;
    char const* argv = "";
    // get around warning:
    // ISO C++11 does not allow conversion from string literal to 'char *' [-Wwritable-strings]
    char* argvv = const_cast<char*>(argv);

    initialize(argc, &argvv);
}

void finalize()
{
    Kokkos::finalize();
    MPI_Finalize();
}
}  // namespace mrmd
