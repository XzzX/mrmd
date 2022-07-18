#include "PeriodicMapping.hpp"

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace PeriodicMapping
{
void mapIntoDomain(data::Atoms& atoms, const data::Subdomain& subdomain)
{
    auto pos = atoms.getPos();
    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            auto& x = pos(idx, dim);
            if (subdomain.maxCorner[dim] <= x)
            {
                x -= subdomain.diameter[dim];
                if (x < subdomain.minCorner[dim])
                {
                    x = subdomain.minCorner[dim];
                }
            }
            if (x < subdomain.minCorner[dim])
            {
                x += subdomain.diameter[dim];
                if (subdomain.maxCorner[dim] <= x)
                {
                    x = subdomain.minCorner[dim];
                }
            }
            assert(x < subdomain.maxCorner[dim]);
            assert(subdomain.minCorner[dim] <= x);
        }
    };
    Kokkos::parallel_for("PeriodicMapping::mapIntoDomain", policy, kernel);
    Kokkos::fence();
}
}  // namespace PeriodicMapping
}  // namespace impl
}  // namespace communication
}  // namespace mrmd