#include "action.hpp"

#include <pybind11/stl.h>

#include <action/BerendsenBarostat.hpp>
#include <action/BerendsenThermostat.hpp>
#include <action/ContributeMoleculeForceToAtoms.hpp>
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
        .def("sample", &action::ThermodynamicForce::sample)
        .def("update", &action::ThermodynamicForce::update)
        .def("apply",
             static_cast<void (action::ThermodynamicForce::*)(const data::Atoms&) const>(
                 &action::ThermodynamicForce::apply))
        .def("apply",
             static_cast<void (action::ThermodynamicForce::*)(
                 const data::Atoms&, const weighting_function::Slab&) const>(
                 &action::ThermodynamicForce::apply));

    auto berendsen_thermostat = m.def_submodule("BerendsenThermostat", "");
    berendsen_thermostat.def("apply", &action::BerendsenThermostat::apply);

    auto berendsen_barostat = m.def_submodule("BerendsenBarostat", "");
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