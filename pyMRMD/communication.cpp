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

#include "communication.hpp"

#include <communication/GhostLayer.hpp>
#include <communication/MultiResGhostLayer.hpp>

namespace py = pybind11;

void init_communication(py::module_& m)
{
    using namespace mrmd;

    py::class_<communication::GhostLayer>(m, "GhostLayer")
        .def(py::init<>())
        .def("exchange_real_atoms", &communication::GhostLayer::exchangeRealAtoms)
        .def("create_ghost_atoms", &communication::GhostLayer::createGhostAtoms)
        .def("update_ghost_atoms", &communication::GhostLayer::updateGhostAtoms)
        .def("contribute_back_ghost_to_real",
             &communication::GhostLayer::contributeBackGhostToReal);

    py::class_<communication::MultiResGhostLayer>(m, "MultiResGhostLayer")
        .def(py::init<>())
        .def("exchange_real_atoms", &communication::MultiResGhostLayer::exchangeRealAtoms)
        .def("create_ghost_atoms", &communication::MultiResGhostLayer::createGhostAtoms)
        .def("update_ghost_atoms", &communication::MultiResGhostLayer::updateGhostAtoms)
        .def("contribute_back_ghost_to_real",
             &communication::MultiResGhostLayer::contributeBackGhostToReal);
}