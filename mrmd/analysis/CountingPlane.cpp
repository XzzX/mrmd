#include "CountingPlane.hpp"

#include "assert/assert.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd::analysis
{
CountingPlane::CountingPlane(const Point3D& pointOnPlane, Vector3D planeNormal)
    : pointOnPlane_(pointOnPlane), planeNormal_(planeNormal), distanceToPlane_("distanceToPlane", 0)
{
}

void CountingPlane::startCounting(data::Atoms& atoms)
{
    auto pos = atoms.getPos();
    util::grow(distanceToPlane_, atoms.size());
    Kokkos::parallel_for(
        "ComputeDistanceToPlane",
        Kokkos::RangePolicy<>(0, atoms.size()),
        KOKKOS_LAMBDA(const idx_t i) {
            distanceToPlane_(i) = (pos(i, 0) - pointOnPlane_[0]) * planeNormal_[0] +
                                  (pos(i, 1) - pointOnPlane_[1]) * planeNormal_[1] +
                                  (pos(i, 2) - pointOnPlane_[2]) * planeNormal_[2];
        });
}

int64_t CountingPlane::stopCounting(data::Atoms& atoms)
{
    auto pos = atoms.getPos();
    MRMD_HOST_CHECK_GREATEREQUAL(distanceToPlane_.size(),
                                 pos.size(),
                                 "You must call startCounting before stopCounting. The number of "
                                 "particles is not allowed to change!");
    int64_t count = 0;
    Kokkos::parallel_reduce(
        "CountCrossings",
        Kokkos::RangePolicy<>(0, atoms.size()),
        KOKKOS_LAMBDA(const idx_t i, int64_t& localCount) {
            auto dist = (pos(i, 0) - pointOnPlane_[0]) * planeNormal_[0] +
                        (pos(i, 1) - pointOnPlane_[1]) * planeNormal_[1] +
                        (pos(i, 2) - pointOnPlane_[2]) * planeNormal_[2];
            if (dist * distanceToPlane_(i) < 0)
            {
                localCount += 1;
            }
        },
        count);
    return count;
}

}  // namespace mrmd::analysis