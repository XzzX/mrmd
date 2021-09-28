#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
class MeanSquareDisplacement
{
private:
    VectorView initialPosition_;
    idx_t numAtoms_;
    data::Subdomain subdomain_;

public:
    /**
     * Store current positions as a starting point.
     */
    void reset(data::Atoms& atoms);
    void reset(data::Molecules& molecules);
    /**
     * Compare current positions against the starting point
     * @return mean squre displacement
     */
    real_t calc(data::Atoms& atoms);
    real_t calc(data::Molecules& molecules);

    MeanSquareDisplacement(const data::Subdomain& subdomain)
        : initialPosition_("MeanSquareDisplacement::initialPosition", 0), subdomain_(subdomain)
    {
    }
};

}  // namespace analysis
}  // namespace mrmd