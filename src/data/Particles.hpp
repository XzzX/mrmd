#pragma once

#include <Cabana_Core.hpp>

#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
class Particles
{
public:
    /// number of spatial dimensions
    constexpr static int DIMENSIONS = 3;
    constexpr static int VECTOR_LENGTH = 1;

    enum Props
    {
        POS = 0,
        VEL = 1,
        FORCE = 2,
        TYPE = 3,
        RELATIVE_MASS = 4,  ///< relative mass of the atom in relation to the molecule
    };
    using DataTypes = Cabana::
        MemberTypes<real_t[DIMENSIONS], real_t[DIMENSIONS], real_t[DIMENSIONS], idx_t, real_t>;
    using ParticlesT = Cabana::AoSoA<DataTypes, DeviceType, VECTOR_LENGTH>;

    using pos_t = typename ParticlesT::template member_slice_type<POS>;
    using vel_t = typename ParticlesT::template member_slice_type<VEL>;
    using force_t = typename ParticlesT::template member_slice_type<FORCE>;
    using type_t = typename ParticlesT::template member_slice_type<TYPE>;
    using relative_mass_t = typename ParticlesT::template member_slice_type<RELATIVE_MASS>;

    pos_t pos;
    vel_t vel;
    force_t force;
    type_t type;
    relative_mass_t relativeMass;

    KOKKOS_FORCEINLINE_FUNCTION pos_t getPos() const { return pos; }
    KOKKOS_FORCEINLINE_FUNCTION vel_t getVel() const { return vel; }
    KOKKOS_FORCEINLINE_FUNCTION force_t getForce() const { return force; }
    KOKKOS_FORCEINLINE_FUNCTION type_t getType() const { return type; }
    KOKKOS_FORCEINLINE_FUNCTION relative_mass_t getRelativeMass() const { return relativeMass; }

    void sliceAll()
    {
        pos = Cabana::slice<POS>(particles_);
        vel = Cabana::slice<VEL>(particles_);
        force = Cabana::slice<FORCE>(particles_);
        type = Cabana::slice<TYPE>(particles_);
        relativeMass = Cabana::slice<RELATIVE_MASS>(particles_);
    }

    KOKKOS_INLINE_FUNCTION auto size() const { return particles_.size(); }
    auto numSoA() const { return particles_.numSoA(); }
    auto arraySize(size_t s) const { return particles_.arraySize(s); }

    void resize(size_t size)
    {
        particles_.resize(size);
        sliceAll();
    }

    KOKKOS_INLINE_FUNCTION
    void permute(LinkedCellList& linkedCellList) const
    {
        Cabana::permute(linkedCellList, particles_);
    }

    KOKKOS_INLINE_FUNCTION
    void copy(const idx_t dst, const idx_t src) const
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            pos(dst, dim) = pos(src, dim);
            vel(dst, dim) = vel(src, dim);
            force(dst, dim) = force(src, dim);
        }
        type(dst) = type(src);
        relativeMass(dst) = relativeMass(src);
    }

    void removeGhostParticles()
    {
        numGhostParticles = 0;
        resize(numLocalParticles + numGhostParticles);
    }

    auto getAoSoA() { return particles_; }

    idx_t numLocalParticles = 0;
    idx_t numGhostParticles = 0;

    explicit Particles(const idx_t numParticles) : particles_("particles", numParticles)
    {
        sliceAll();
    }

private:
    ParticlesT particles_;
};
}  // namespace data
}  // namespace mrmd