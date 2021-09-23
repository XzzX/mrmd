#pragma once

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class PeriodicMapping
{
private:
    data::Atoms::pos_t pos_;
    const data::Subdomain subdomain_;

public:
    /**
     *
     * @pre Atom position is at most one periodic copy away
     * from the subdomain.
     * @post Atom position lies within half-open interval
     * [min, max) for all coordinate dimensions.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            auto& x = pos_(idx, dim);
            if (subdomain_.maxCorner[dim] <= x)
            {
                x -= subdomain_.diameter[dim];
                if (x < subdomain_.minCorner[dim])
                {
                    x = subdomain_.minCorner[dim];
                }
            }
            if (x < subdomain_.minCorner[dim])
            {
                x += subdomain_.diameter[dim];
                if (subdomain_.maxCorner[dim] <= x)
                {
                    x = subdomain_.minCorner[dim];
                }
            }
            assert(x < subdomain_.maxCorner[dim]);
            assert(subdomain_.minCorner[dim] <= x);
        }
    }

    void mapIntoDomain(data::Atoms& atoms)
    {
        pos_ = atoms.getPos();
        auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
        Kokkos::parallel_for(policy, *this, "PeriodicMapping::mapIntoDomain");
        Kokkos::fence();
    }

    PeriodicMapping(const data::Subdomain& subdomain) : subdomain_(subdomain) {}
};

}  // namespace impl
}  // namespace communication
}  // namespace mrmd