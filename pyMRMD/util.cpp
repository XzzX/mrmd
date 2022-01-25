#include "util.hpp"

#include <Kokkos_Core.hpp>

namespace py = pybind11;

void init_util(py::module_ &m)
{
    py::class_<Kokkos::Timer>(m, "Timer")
        .def(py::init<>())
        .def("reset", &Kokkos::Timer::reset)
        .def("seconds", &Kokkos::Timer::seconds);
}