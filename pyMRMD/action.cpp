#include "action.hpp"

#include <pybind11/stl.h>

#include <action/LangevinThermostat.hpp>
#include <action/LennardJones.hpp>
#include <action/VelocityVerlet.hpp>

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

    auto vv = m.def_submodule("velocity_verlet", "");
    vv.def("pre_force_integrate", &action::VelocityVerlet::preForceIntegrate);
    vv.def("post_force_integrate", &action::VelocityVerlet::postForceIntegrate);
}