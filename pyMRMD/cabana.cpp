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

#include "cabana.hpp"

#include <data/Atoms.hpp>
#include <data/Subdomain.hpp>
#include <datatypes.hpp>

namespace py = pybind11;

void init_cabana(py::module_ &m)
{
    using size_type = Kokkos::DefaultExecutionSpace::memory_space::size_type;

    using real3_t = mrmd::data::Atoms::pos_t;
    py::class_<real3_t>(m, "real3_t", py::buffer_protocol())
        .def_buffer(
            [](real3_t &m) -> py::buffer_info
            {
                return py::buffer_info(
                    m.data(),                                             /* Pointer to buffer */
                    sizeof(real3_t::value_type),                          /* Size of one scalar */
                    py::format_descriptor<real3_t::value_type>::format(), /* Python struct-style
                                                                           * format descriptor
                                                                           */
                    3,                                                    /* Number of dimensions */
                    {m.extent(2), m.extent(1), m.extent(0)},              /* Buffer dimensions */
                    {sizeof(real3_t::value_type) * m.stride(2),
                     sizeof(real3_t::value_type) *
                         m.stride(1), /* Strides (in bytes) for each index */
                     sizeof(real3_t::value_type) * m.stride(0)});
            })
        .def(
            "access",
            [](const real3_t &cls, size_type idx, size_type dim) { return cls(idx, dim); },
            py::return_value_policy::reference);

    using real1_t = mrmd::data::Atoms::mass_t;
    py::class_<real1_t>(m, "real1_t", py::buffer_protocol())
        .def_buffer(
            [](real1_t &m) -> py::buffer_info
            {
                return py::buffer_info(
                    m.data(),                                             /* Pointer to buffer */
                    sizeof(real1_t::value_type),                          /* Size of one scalar */
                    py::format_descriptor<real1_t::value_type>::format(), /* Python struct-style
                                                                           * format descriptor
                                                                           */
                    2,                                                    /* Number of dimensions */
                    {m.extent(1), m.extent(0)},                           /* Buffer dimensions */
                    {sizeof(real1_t::value_type) *
                         m.stride(1), /* Strides (in bytes) for each index */
                     sizeof(real1_t::value_type) * m.stride(0)});
            })
        .def(
            "access",
            [](const real1_t &cls, size_type idx) { return cls(idx); },
            py::return_value_policy::reference);

    using idx1_t = mrmd::data::Atoms::type_t;
    py::class_<idx1_t>(m, "idx1_t", py::buffer_protocol())
        .def_buffer(
            [](idx1_t &m) -> py::buffer_info
            {
                return py::buffer_info(
                    m.data(),                                            /* Pointer to buffer */
                    sizeof(idx1_t::value_type),                          /* Size of one scalar */
                    py::format_descriptor<idx1_t::value_type>::format(), /* Python struct-style
                                                                          * format descriptor
                                                                          */
                    2,                                                   /* Number of dimensions */
                    {m.extent(1), m.extent(0)},                          /* Buffer dimensions */
                    {sizeof(idx1_t::value_type) *
                         m.stride(1), /* Strides (in bytes) for each index */
                     sizeof(idx1_t::value_type) * m.stride(0)});
            })
        .def(
            "access",
            [](const idx1_t &cls, size_type idx) { return cls(idx); },
            py::return_value_policy::reference);

    py::class_<mrmd::HalfVerletList>(m, "HalfVerletList")
        .def(py::init<>())
        .def("build", &mrmd::HalfVerletList::build<mrmd::data::Atoms::pos_t>);

    py::class_<mrmd::FullVerletList>(m, "FullVerletList")
        .def(py::init<>())
        .def("build", &mrmd::FullVerletList::build<mrmd::data::Atoms::pos_t>);

    using namespace mrmd;
    m.def("build_verlet_list",
          [](HalfVerletList &verletList,
             const data::Atoms &atoms,
             const data::Subdomain &subdomain,
             real_t neighborCutoff,
             real_t cellRatio,
             idx_t estimatedMaxNeighbors)
          {
              verletList.build(atoms.getPos(),
                               0,
                               atoms.numLocalAtoms,
                               neighborCutoff,
                               cellRatio,
                               subdomain.minGhostCorner.data(),
                               subdomain.maxGhostCorner.data(),
                               estimatedMaxNeighbors);
          });
    m.def("build_verlet_list",
          [](FullVerletList &verletList,
             const data::Atoms &atoms,
             const data::Subdomain &subdomain,
             real_t neighborCutoff,
             real_t cellRatio,
             idx_t estimatedMaxNeighbors)
          {
              verletList.build(atoms.getPos(),
                               0,
                               atoms.numLocalAtoms,
                               neighborCutoff,
                               cellRatio,
                               subdomain.minGhostCorner.data(),
                               subdomain.maxGhostCorner.data(),
                               estimatedMaxNeighbors);
          });
}