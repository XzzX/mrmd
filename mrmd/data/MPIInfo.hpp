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

#pragma once

#include "mpi_helper.hpp"

namespace mrmd::data
{
struct MPIInfo
{
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;
    int size = -1;

    explicit MPIInfo(MPI_Comm communicator = MPI_COMM_WORLD)
    {
        comm = communicator;
        CHECK_MPI(MPI_Comm_rank(communicator, &rank));
        CHECK_MPI(MPI_Comm_size(communicator, &size));
    }
};
}  // namespace mrmd::data