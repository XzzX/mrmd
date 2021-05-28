#pragma once

#include <Cabana_Core.hpp>

class Particles{
public:
    constexpr static int dim = 3;

    auto getPos() {return Cabana::slice<POS>(particles_);}
    auto getVel() {return Cabana::slice<VEL>(particles_);}
    auto getForce() {return Cabana::slice<FORCE>(particles_);}

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
    constexpr static int VectorLength = 8;
    using ParticlesT = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

    ParticlesT particles_ = ParticlesT("particles", 100);
};