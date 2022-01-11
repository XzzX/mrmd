#include "MPIInfo.hpp"

#include <gtest/gtest.h>

#include "datatypes.hpp"

namespace mrmd::data
{
TEST(MPIInfo, Rank)
{
    MPIInfo mpiInfo(MPI_COMM_WORLD);
    int sumRanks = 0;
    MPI_Allreduce(&mpiInfo.rank, &sumRanks, 1, MPI_INT, MPI_SUM, mpiInfo.comm);
    EXPECT_EQ(sumRanks, 1);
}
TEST(MPIInfo, Size)
{
    MPIInfo mpiInfo(MPI_COMM_WORLD);
    EXPECT_EQ(mpiInfo.size, 2);
}

}  // namespace mrmd::data