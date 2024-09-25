// Copyright 2024 Sebastian Eibl
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "analysis.hpp"

#include <analysis/AxialDensityProfile.hpp>
#include <analysis/Fluctuation.hpp>
#include <analysis/KineticEnergy.hpp>
#include <analysis/MeanSquareDisplacement.hpp>
#include <analysis/Pressure.hpp>

namespace py = pybind11;

void init_analysis(py::module_ &m)
{
    using namespace mrmd;

    m.def("get_fluctuation", &analysis::getFluctuation);
    m.def("get_kinetic_energy", &analysis::getKineticEnergy);
    m.def("get_mean_kinetic_energy", &analysis::getMeanKineticEnergy);
    m.def("get_pressure", &analysis::getPressure);
    m.def("get_axial_density_profile",
          [](const idx_t numAtoms,
             const data::Atoms &atoms,
             const int64_t numTypes,
             const real_t min,
             const real_t max,
             const int64_t numBins,
             const int64_t axis)
          {
              return analysis::getAxialDensityProfile(
                  numAtoms, atoms.getPos(), atoms.getType(), numTypes, min, max, numBins, axis);
          });

    py::class_<analysis::MeanSquareDisplacement>(m, "MeanSquareDisplacement")
        .def(py::init<>())
        .def("calc",
             py::overload_cast<data::Atoms &, const data::Subdomain &>(
                 &analysis::MeanSquareDisplacement::calc))
        .def("reset", py::overload_cast<data::Atoms &>(&analysis::MeanSquareDisplacement::reset));
}