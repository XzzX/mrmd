#pragma once

#include <Cabana_Core.hpp>

class Particles{
public:
    constexpr static int dim = 3;
    constexpr static int VectorLength = 8;

    enum Props {
        POS = 0,
        VEL = 1,
        FORCE = 2
    };
    using DeviceType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    using DataTypes = Cabana::MemberTypes<double[dim], double[dim], double[dim]>;
    using ParticlesT = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

    using pos_t = typename ParticlesT::template member_slice_type<POS>;
    using vel_t = typename ParticlesT::template member_slice_type<VEL>;
    using force_t = typename ParticlesT::template member_slice_type<FORCE>;

    pos_t getPos() {return Cabana::slice<POS>(particles_);}
    vel_t getVel() {return Cabana::slice<VEL>(particles_);}
    force_t getForce() {return Cabana::slice<FORCE>(particles_);}

    auto size() const {return particles_.size();}
    auto numSoA() const {return particles_.numSoA();}
    auto arraySize(size_t s) const {return particles_.arraySize(s);}

    void resize(size_t size)
    {
        particles_.resize(size);
    }
private:

    ParticlesT particles_ = ParticlesT("particles", 40000);
};