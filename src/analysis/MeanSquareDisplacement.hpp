#pragma once

#include "data/Atoms.hpp"
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
    void reset(data::Atoms& atoms);
    real_t calc(data::Atoms& atoms);

    MeanSquareDisplacement(const data::Subdomain& subdomain)
        : initialPosition_("MeanSquareDisplacement::initialPosition", 0), subdomain_(subdomain)
    {
    }
};

}  // namespace analysis
}  // namespace mrmd