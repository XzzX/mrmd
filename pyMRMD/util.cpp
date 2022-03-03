#include "util.hpp"

#include <Kokkos_Core.hpp>
#include <util/ExponentialMovingAverage.hpp>

namespace py = pybind11;

void init_util(py::module_ &m)
{
    using namespace mrmd;

    py::class_<util::ExponentialMovingAverage>(m, "ExponentialMovingAverage")
        .def(py::init<const real_t &>())
        .def("append", &util::ExponentialMovingAverage::append)
        .def("toReal", &util::ExponentialMovingAverage::toReal);

    py::class_<Kokkos::Timer>(m, "Timer")
        .def(py::init<>())
        .def("reset", &Kokkos::Timer::reset)
        .def("seconds", &Kokkos::Timer::seconds);
}