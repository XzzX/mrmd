#pragma once

#include <Cabana_Core.hpp>

#include "cmake.hpp"
#include "constants.hpp"
#include "datatypes.hpp"

namespace mrmd::data
{
namespace impl
{
/**
 * Data structure for storing information about atoms on the host and the device.
 * @tparam DEVICE_TYPE memory location, \a HostType or \a DeviceType
 * @tparam DEVICE artificial parameter to distinguish classes if HostType==DeviceType
 * for CPU backends
 */
template <class DEVICE_TYPE = DeviceType, bool DEVICE = true>
class GeneralAtoms
{
public:
    enum Props
    {
        POS = 0,
        VEL = 1,
        FORCE = 2,
        TYPE = 3,
        MASS = 4,
        CHARGE = 5,
        RELATIVE_MASS = 6,  ///< relative mass of the atom in relation to the molecule
    };
    using DataTypes = Cabana::MemberTypes<real_t[DIMENSIONS],
                                          real_t[DIMENSIONS],
                                          real_t[DIMENSIONS],
                                          idx_t,
                                          real_t,
                                          real_t,
                                          real_t>;
    using AtomsT = Cabana::AoSoA<DataTypes, typename DEVICE_TYPE::memory_space, VECTOR_LENGTH>;

    using pos_t = typename AtomsT::template member_slice_type<POS>;
    using vel_t = typename AtomsT::template member_slice_type<VEL>;
    using force_t = typename AtomsT::template member_slice_type<FORCE>;
    using type_t = typename AtomsT::template member_slice_type<TYPE>;
    using mass_t = typename AtomsT::template member_slice_type<MASS>;
    using charge_t = typename AtomsT::template member_slice_type<CHARGE>;
    using relative_mass_t = typename AtomsT::template member_slice_type<RELATIVE_MASS>;

    KOKKOS_FORCEINLINE_FUNCTION pos_t getPos() const { return pos; }
    KOKKOS_FORCEINLINE_FUNCTION vel_t getVel() const { return vel; }
    KOKKOS_FORCEINLINE_FUNCTION force_t getForce() const { return force; }
    KOKKOS_FORCEINLINE_FUNCTION type_t getType() const { return type; }
    KOKKOS_FORCEINLINE_FUNCTION charge_t getMass() const { return mass; }
    KOKKOS_FORCEINLINE_FUNCTION charge_t getCharge() const { return charge; }
    KOKKOS_FORCEINLINE_FUNCTION relative_mass_t getRelativeMass() const { return relativeMass; }

    void setForce(const real_t& val) const { Cabana::deep_copy(force, val); }

    void sliceAll()
    {
        pos = Cabana::slice<POS>(atoms_);
        vel = Cabana::slice<VEL>(atoms_);
        force = Cabana::slice<FORCE>(atoms_);
        type = Cabana::slice<TYPE>(atoms_);
        mass = Cabana::slice<MASS>(atoms_);
        charge = Cabana::slice<CHARGE>(atoms_);
        relativeMass = Cabana::slice<RELATIVE_MASS>(atoms_);
    }

    KOKKOS_INLINE_FUNCTION auto size() const { return atoms_.size(); }
    KOKKOS_INLINE_FUNCTION auto numSoA() const { return atoms_.numSoA(); }
    KOKKOS_INLINE_FUNCTION auto arraySize(size_t s) const { return atoms_.arraySize(s); }

    void resize(size_t size)
    {
        atoms_.resize(size);
        sliceAll();
    };

    void permute(LinkedCellList& linkedCellList) const { Cabana::permute(linkedCellList, atoms_); }

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
        mass(dst) = mass(src);
        charge(dst) = charge(src);
        relativeMass(dst) = relativeMass(src);
    }

    void removeGhostAtoms()
    {
        numGhostAtoms = 0;
        resize(numLocalAtoms + numGhostAtoms);
    }

    auto& getAoSoA() const { return atoms_; }

    idx_t numLocalAtoms = 0;
    idx_t numGhostAtoms = 0;

    explicit GeneralAtoms(const idx_t numAtoms) : atoms_("atoms", numAtoms)
    {
        sliceAll();
        Cabana::deep_copy(pos, 0_r);
        Cabana::deep_copy(vel, 0_r);
        Cabana::deep_copy(force, 0_r);
        Cabana::deep_copy(type, 0);
        Cabana::deep_copy(mass, 0_r);
        Cabana::deep_copy(charge, 0_r);
        Cabana::deep_copy(relativeMass, 0_r);
    }
    template <class DEVICE_TYPE_SRC, bool DEVICE_SRC>
    explicit GeneralAtoms(const GeneralAtoms<DEVICE_TYPE_SRC, DEVICE_SRC>& atoms);

private:
    AtomsT atoms_;

    pos_t pos;
    vel_t vel;
    force_t force;
    type_t type;
    mass_t mass;
    charge_t charge;
    relative_mass_t relativeMass;
};
}  // namespace impl

template <class DEVICE_TYPE_A, bool DEVICE_A, class DEVICE_TYPE_B, bool DEVICE_B>
void deep_copy(impl::GeneralAtoms<DEVICE_TYPE_A, DEVICE_A>& dst,
               const impl::GeneralAtoms<DEVICE_TYPE_B, DEVICE_B>& src)
{
    dst.numLocalAtoms = src.numLocalAtoms;
    dst.numGhostAtoms = src.numGhostAtoms;
    dst.resize(src.size());
    Cabana::deep_copy(dst.getAoSoA(), src.getAoSoA());
    dst.sliceAll();
}

namespace impl
{
template <class DEVICE_TYPE, bool DEVICE>
template <class DEVICE_TYPE_SRC, bool DEVICE_SRC>
GeneralAtoms<DEVICE_TYPE, DEVICE>::GeneralAtoms(
    const GeneralAtoms<DEVICE_TYPE_SRC, DEVICE_SRC>& atoms)
    : atoms_("atoms", atoms.size())
{
    deep_copy(*this, atoms);
}
}  // namespace impl

/// atoms with memory allocated on host
using HostAtoms = impl::GeneralAtoms<HostType, false>;
/// atoms with memory allocated on device
using DeviceAtoms = impl::GeneralAtoms<DeviceType, true>;
using Atoms = DeviceAtoms;
}  // namespace mrmd::data