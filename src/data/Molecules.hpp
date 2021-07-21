#pragma once

#include <Cabana_Core.hpp>

#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
class Molecules
{
public:
    /// number of spatial dimensions
    constexpr static int DIMENSIONS = 3;
    constexpr static int VECTOR_LENGTH = 1;

    enum Props
    {
        POS = 0,
        LAMBDA = 1,
        GRAD_LAMBDA = 2,
        ATOMS_END_IDX = 3,  ///< exclusive end offset of atoms
    };
    using DataTypes = Cabana::MemberTypes<real_t[DIMENSIONS], real_t, real_t, idx_t>;
    using MoleculesT = Cabana::AoSoA<DataTypes, DeviceType, VECTOR_LENGTH>;

    using pos_t = typename MoleculesT::template member_slice_type<POS>;
    using lambda_t = typename MoleculesT::template member_slice_type<LAMBDA>;
    using grad_lambda_t = typename MoleculesT::template member_slice_type<GRAD_LAMBDA>;
    using atoms_end_idx_t = typename MoleculesT::template member_slice_type<ATOMS_END_IDX>;

    pos_t pos;
    lambda_t lambda;
    grad_lambda_t gradLambda;
    atoms_end_idx_t atomsEndIdx;

    KOKKOS_FORCEINLINE_FUNCTION pos_t getPos() const { return pos; }
    KOKKOS_FORCEINLINE_FUNCTION lambda_t getLambda() const { return lambda; }
    KOKKOS_FORCEINLINE_FUNCTION grad_lambda_t getGradLambda() const { return gradLambda; }
    KOKKOS_FORCEINLINE_FUNCTION atoms_end_idx_t getAtomsEndIdx() const { return atomsEndIdx; }

    void sliceAll()
    {
        pos = Cabana::slice<POS>(molecules_);
        lambda = Cabana::slice<LAMBDA>(molecules_);
        gradLambda = Cabana::slice<GRAD_LAMBDA>(molecules_);
        atomsEndIdx = Cabana::slice<ATOMS_END_IDX>(molecules_);
    }

    KOKKOS_INLINE_FUNCTION auto size() const { return molecules_.size(); }
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
        }
        lambda(dst) = lambda(src);
        gradLambda(dst) = gradLambda(src);
        atomsEndIdx(dst) = atomsEndIdx(src);
    }

    void removeGhostMolecules()
    {
        numGhostMolecules = 0;
        resize(numLocalMolecules + numGhostMolecules);
    }

    auto getAoSoA() { return molecules_; }

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