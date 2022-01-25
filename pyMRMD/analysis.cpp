#include "analysis.hpp"

#include <analysis/KineticEnergy.hpp>
#include <analysis/MeanSquareDisplacement.hpp>
#include <analysis/Pressure.hpp>

namespace py = pybind11;

void init_analysis(py::module_ &m)
{
    using namespace mrmd;

    m.def("get_kinetic_energy", &analysis::getKineticEnergy);
    m.def("get_mean_kinetic_energy", &analysis::getMeanKineticEnergy);
    m.def("get_pressure", &analysis::getPressure);

    py::class_<analysis::MeanSquareDisplacement>(m, "MeanSquareDisplacement")
        .def(py::init<>())
        .def("calc",
             py::overload_cast<data::Atoms &, const data::Subdomain &>(
                 &analysis::MeanSquareDisplacement::calc))
        .def("reset", py::overload_cast<data::Atoms &>(&analysis::MeanSquareDisplacement::reset));
}