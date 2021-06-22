#pragma once

#include <Cabana_Core.hpp>

#include "datatypes.hpp"

class Particles
{
public:
    constexpr static int dim = 3;
    constexpr static int VectorLength = 8;

    enum Props
    {
        POS = 0,
        VEL = 1,
        FORCE = 2,
        GHOST = 3
    };
    using DeviceType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    using DataTypes = Cabana::MemberTypes<double[dim], double[dim], double[dim], idx_t>;
    using ParticlesT = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

    using pos_t = typename ParticlesT::template member_slice_type<POS>;
    using vel_t = typename ParticlesT::template member_slice_type<VEL>;
    using force_t = typename ParticlesT::template member_slice_type<FORCE>;
    using ghost_t = typename ParticlesT::template member_slice_type<GHOST>;

    pos_t getPos() { return Cabana::slice<POS>(particles_); }
    vel_t getVel() { return Cabana::slice<VEL>(particles_); }
    force_t getForce() { return Cabana::slice<FORCE>(particles_); }
    ghost_t getGhost() { return Cabana::slice<GHOST>(particles_); }

    auto size() const { return particles_.size(); }
    auto numSoA() const { return particles_.numSoA(); }
    auto arraySize(size_t s) const { return particles_.arraySize(s); }

    void resize(size_t size) { particles_.resize(size); }

    void copy(const idx_t src, const idx_t dst);

    void removeGhostParticles()
    {
        numGhostParticles = 0;
        resize(numLocalParticles + numGhostParticles);
    }

    auto getAoSoA() { return particles_; }

    idx_t numLocalParticles = 0;
    idx_t numGhostParticles = 0;

private:
    ParticlesT particles_ = ParticlesT("particles", 100000);
};