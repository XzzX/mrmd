#pragma once

#include <Cabana_Core.hpp>

#include "cmake.hpp"
#include "constants.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
class Molecules
{
public:
    enum Props
    {
        POS = 0,
        FORCE = 1,
        LAMBDA = 2,
        MODULATED_LAMBDA = 3,
        GRAD_LAMBDA = 4,
        ATOMS_OFFSET = 5,  ///< offset into atoms array
        NUM_ATOMS = 6,     ///< number of atoms for this molecule
    };
    using DataTypes = Cabana::MemberTypes<real_t[DIMENSIONS],
                                          real_t[DIMENSIONS],
                                          real_t,
                                          real_t,
                                          real_t[DIMENSIONS],
                                          idx_t,
                                          idx_t>;
    using MoleculesT = Cabana::AoSoA<DataTypes, DeviceType, VECTOR_LENGTH>;

    using pos_t = typename MoleculesT::template member_slice_type<POS>;
    using force_t = typename MoleculesT::template member_slice_type<FORCE>;
    using lambda_t = typename MoleculesT::template member_slice_type<LAMBDA>;
    using modulated_lambda_t = typename MoleculesT::template member_slice_type<MODULATED_LAMBDA>;
    using grad_lambda_t = typename MoleculesT::template member_slice_type<GRAD_LAMBDA>;
    using atoms_offset_t = typename MoleculesT::template member_slice_type<ATOMS_OFFSET>;
    using num_atoms_t = typename MoleculesT::template member_slice_type<NUM_ATOMS>;

    pos_t pos;
    force_t force;
    lambda_t lambda;
    modulated_lambda_t modulatedLambda;
    grad_lambda_t gradLambda;
    atoms_offset_t atomsOffset;
    num_atoms_t numAtoms;

    KOKKOS_FORCEINLINE_FUNCTION pos_t getPos() const { return pos; }
    KOKKOS_FORCEINLINE_FUNCTION force_t getForce() const { return force; }
    KOKKOS_FORCEINLINE_FUNCTION lambda_t getLambda() const { return lambda; }
    KOKKOS_FORCEINLINE_FUNCTION modulated_lambda_t getModulatedLambda() const
    {
        return modulatedLambda;
    }
    KOKKOS_FORCEINLINE_FUNCTION grad_lambda_t getGradLambda() const { return gradLambda; }
    KOKKOS_FORCEINLINE_FUNCTION atoms_offset_t getAtomsOffset() const { return atomsOffset; }
    KOKKOS_FORCEINLINE_FUNCTION num_atoms_t getNumAtoms() const { return numAtoms; }

    void sliceAll()
    {
        pos = Cabana::slice<POS>(molecules_);
        force = Cabana::slice<FORCE>(molecules_);
        lambda = Cabana::slice<LAMBDA>(molecules_);
        modulatedLambda = Cabana::slice<MODULATED_LAMBDA>(molecules_);
        gradLambda = Cabana::slice<GRAD_LAMBDA>(molecules_);
        atomsOffset = Cabana::slice<ATOMS_OFFSET>(molecules_);
        numAtoms = Cabana::slice<NUM_ATOMS>(molecules_);
    }

    KOKKOS_INLINE_FUNCTION idx_t size() const { return idx_c(molecules_.size()); }
    auto numSoA() const { return molecules_.numSoA(); }
    auto arraySize(size_t s) const { return molecules_.arraySize(s); }

    void resize(size_t size)
    {
        molecules_.resize(size);
        sliceAll();
    }

    KOKKOS_INLINE_FUNCTION
    void permute(LinkedCellList& linkedCellList) const
    {
        Cabana::permute(linkedCellList, molecules_);
    }

    KOKKOS_INLINE_FUNCTION
    void copy(const idx_t dst, const idx_t src) const
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            pos(dst, dim) = pos(src, dim);
            force(dst, dim) = force(src, dim);
            gradLambda(dst, dim) = gradLambda(src, dim);
        }
        lambda(dst) = lambda(src);
        modulatedLambda(dst) = modulatedLambda(src);
        atomsOffset(dst) = atomsOffset(src);
        numAtoms(dst) = numAtoms(src);
    }

    void removeGhostMolecules()
    {
        numGhostMolecules = 0;
        resize(numLocalMolecules + numGhostMolecules);
    }

    auto& getAoSoA() { return molecules_; }

    idx_t numLocalMolecules = 0;
    idx_t numGhostMolecules = 0;

    explicit Molecules(const idx_t numMolecules) : molecules_("molecules", numMolecules)
    {
        sliceAll();
    }

private:
    MoleculesT molecules_;
};
}  // namespace data
}  // namespace mrmd