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

#include "weighting_function.hpp"

#include <pybind11/stl.h>

#include <datatypes.hpp>
#include <weighting_function/Slab.hpp>

namespace py = pybind11;

void init_weighting_function(py::module_ &m)
{
    using namespace mrmd;
    py::class_<weighting_function::Slab>(m, "Slab")
        .def(py::init<Point3D &, real_t, real_t, idx_t>())
        .def("is_in_at_region", &weighting_function::Slab::isInATRegion)
        .def("is_in_hy_region", &weighting_function::Slab::isInHYRegion)
        .def("is_in_cg_region", &weighting_function::Slab::isInCGRegion);
}