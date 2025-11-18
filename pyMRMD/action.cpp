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

#include "action.hpp"

#include <pybind11/stl.h>

#include <action/BerendsenBarostat.hpp>
#include <action/BerendsenThermostat.hpp>
#include <action/ContributeMoleculeForceToAtoms.hpp>
#include <action/LJ_IdealGas.hpp>
#include <action/LangevinThermostat.hpp>
#include <action/LennardJones.hpp>
#include <action/ThermodynamicForce.hpp>
#include <action/UpdateMolecules.hpp>
#include <action/VelocityVerlet.hpp>
#include <weighting_function/Slab.hpp>
#include <weighting_function/Spherical.hpp>

namespace py = pybind11;

void init_action(py::module_& m)
{
    using namespace mrmd;

    py::class_<action::LangevinThermostat>(m, "LangevinThermostat")
        .def(py::init<real_t, real_t, real_t>())
        .def("get_pref1", &action::LangevinThermostat::getPref1)
        .def("get_pref2", &action::LangevinThermostat::getPref2)
        .def("apply", &action::LangevinThermostat::apply)
        .def("set", &action::LangevinThermostat::set);

    py::class_<action::LennardJones>(m, "LennardJones")
        .def(py::init<real_t&, real_t&, real_t&, real_t&>())
        .def(py::init<const std::vector<real_t>&,
                      const std::vector<real_t>&,
                      const std::vector<real_t>&,
                      const std::vector<real_t>&,
                      const idx_t&,
                      const bool>())
        .def("get_energy", &action::LennardJones::getEnergy)
        .def("get_virial", &action::LennardJones::getVirial)
        .def("apply", &action::LennardJones::apply);

    py::class_<action::LJ_IdealGas>(m, "LJ_IdealGas")
        .def("set_compensation_energy_sampling_interval",
             &action::LJ_IdealGas::setCompensationEnergySamplingInterval)
        .def("set_compensation_energy_update_interval",
             &action::LJ_IdealGas::setCompensationEnergyUpdateInterval)
        .def("get_mean_compensation_energy", &action::LJ_IdealGas::getMeanCompensationEnergy)
        .def("run", &action::LJ_IdealGas::run)
        .def(py::init<const real_t&, const real_t&, const real_t&, const real_t&, const bool>())
        .def(py::init<const std::vector<real_t>&,
                      const std::vector<real_t>&,
                      const std::vector<real_t>&,
                      const std::vector<real_t>&,
                      const idx_t,
                      const bool>());

    py::class_<action::ThermodynamicForce>(m, "ThermodynamicForce")
        .def(py::init<const std::vector<real_t>&,
                      const data::Subdomain&,
                      const real_t&,
                      const std::vector<real_t>&,
                      const bool,
                      const bool>())
        .def(py::init<const real_t,
                      const data::Subdomain&,
                      const real_t&,
                      const real_t,
                      const bool,
                      const bool>())
        .def("get_density_profile",
             static_cast<data::MultiHistogram (action::ThermodynamicForce::*)() const>(
                 &action::ThermodynamicForce::getDensityProfile))
        .def("sample", &action::ThermodynamicForce::sample)
        .def("update", &action::ThermodynamicForce::update)
        .def("apply",
             static_cast<void (action::ThermodynamicForce::*)(const data::Atoms&) const>(
                 &action::ThermodynamicForce::apply))
        .def("get_mu_left", &action::ThermodynamicForce::getMuLeft)
        .def("get_mu_right", &action::ThermodynamicForce::getMuRight);

    auto berendsen_thermostat = m.def_submodule("berendsen_thermostat", "");
    berendsen_thermostat.def("apply", &action::BerendsenThermostat::apply);

    auto berendsen_barostat = m.def_submodule("berendsen_barostat", "");
    berendsen_barostat.def("apply", &action::BerendsenBarostat::apply);

    auto contrib = m.def_submodule("contribute_molecule_force_to_atoms", "");
    contrib.def("update", &action::ContributeMoleculeForceToAtoms::update);

    auto um = m.def_submodule("update_molecules", "");
    um.def("update", &action::UpdateMolecules::update<weighting_function::Slab>);
    //    um.def("update", &action::UpdateMolecules::update<weighting_function::Spherical>);

    auto vv = m.def_submodule("velocity_verlet", "");
    vv.def("pre_force_integrate", &action::VelocityVerlet::preForceIntegrate);
    vv.def("post_force_integrate", &action::VelocityVerlet::postForceIntegrate);
}