#include "communication.hpp"

#include <communication/GhostLayer.hpp>

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
}