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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <datatypes.hpp>
#include <initialization.hpp>

#include "action.hpp"
#include "analysis.hpp"
#include "cabana.hpp"
#include "communication.hpp"
#include "data.hpp"
#include "io.hpp"
#include "util.hpp"
#include "weighting_function.hpp"

namespace py = pybind11;
PYBIND11_MODULE(pyMRMD, m)
{
    using namespace mrmd;

    m.doc() = "MRMD Python Wrapper";

    m.def("initialize", py::overload_cast<>(&mrmd::initialize), "");
    m.def("finalize", &mrmd::finalize, "");

    m.attr("COORD_X") = COORD_X;
    m.attr("COORD_Y") = COORD_Y;
    m.attr("COORD_Z") = COORD_Z;

    py::class_<Point3D>(m, "Point3D", py::buffer_protocol())
        .def_buffer(
            [](Point3D &m) -> py::buffer_info
            {
                return py::buffer_info(
                    m.data(),                                /* Pointer to buffer */
                    sizeof(real_t),                          /* Size of one scalar */
                    py::format_descriptor<real_t>::format(), /* Python struct-style format
                                                              * descriptor
                                                              */
                    1,                                       /* Number of dimensions */
                    {3},                                     /* Buffer dimensions */
                    {sizeof(real_t)}                         /* Strides (in bytes) for each index */
                );
            })
        .def(py::init(
            [](const py::buffer& b)
            {
                /* Request a buffer descriptor from Python */
                py::buffer_info info = b.request();

                /* Some basic validation checks ... */
                if (info.format != py::format_descriptor<real_t>::format())
                    throw std::runtime_error("Wrong floating point type!");

                if (info.ndim != 1) throw std::runtime_error("Incompatible buffer dimension!");

                if (info.strides[0] != sizeof(real_t))
                    throw std::runtime_error("Incompatible buffer stride!");

                auto* data = static_cast<real_t *>(info.ptr);
                return Point3D{data[0], data[1], data[2]};
            }))
        .def(py::init([](real_t x, real_t y, real_t z) { return Point3D{x, y, z}; }))
        .def(py::init(
            [](const py::list& my_list)
            {
                return Point3D{my_list[0].cast<real_t>(),
                               my_list[1].cast<real_t>(),
                               my_list[2].cast<real_t>()};
            }))
        .def("__getitem__", [](Point3D &self, unsigned index) { return self[index]; });
    ;

    py::implicitly_convertible<py::list, Point3D>();

    auto action = m.def_submodule("action", "");
    init_action(action);

    auto analysis = m.def_submodule("analysis", "");
    init_analysis(analysis);

    auto cabana = m.def_submodule("cabana", "");
    init_cabana(cabana);

    auto communication = m.def_submodule("communication", "");
    init_communication(communication);

    auto data = m.def_submodule("data", "");
    init_data(data);

    auto io = m.def_submodule("io", "");
    init_io(io);

    auto util = m.def_submodule("util", "");
    init_util(util);

    auto weighting_function = m.def_submodule("weighting_function", "");
    init_weighting_function(weighting_function);
}