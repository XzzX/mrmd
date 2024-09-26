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
        .def("to_real", &util::ExponentialMovingAverage::toReal);

    py::class_<Kokkos::Timer>(m, "Timer")
        .def(py::init<>())
        .def("reset", &Kokkos::Timer::reset)
        .def("seconds", &Kokkos::Timer::seconds);
}