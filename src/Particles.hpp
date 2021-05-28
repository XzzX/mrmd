#pragma once

#include <Cabana_Core.hpp>

class Particles{
public:
    constexpr static int dim = 3;
    constexpr static int VectorLength = 8;

    auto getPos() {return Cabana::slice<POS>(particles_);}
    auto getVel() {return Cabana::slice<VEL>(particles_);}
    auto getForce() {return Cabana::slice<FORCE>(particles_);}

    auto size() const {return particles_.size();}
    auto numSoA() const {return particles_.numSoA();}
    auto arraySize(size_t s) const {return particles_.arraySize(s);}
private:
    using DeviceType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;

    enum Props {
        POS = 0,
        VEL = 1,
        FORCE = 2
    };
    using DataTypes = Cabana::MemberTypes<double[dim], double[dim], double[dim]>;
    using ParticlesT = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

    ParticlesT particles_ = ParticlesT("particles", 100);
};