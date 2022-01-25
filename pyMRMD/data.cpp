#include "data.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <data/Atoms.hpp>
#include <data/MPIInfo.hpp>
#include <data/Subdomain.hpp>

namespace py = pybind11;

template <class ATOMS_T>
void init_atoms(py::module_ &m, const char *className)
{
    using namespace mrmd;

    py::class_<ATOMS_T>(m, className)
        .def(py::init<const idx_t>())
        .def("get_pos", &ATOMS_T::getPos)
        .def("get_vel", &ATOMS_T::getVel)
        .def("get_force", &ATOMS_T::getForce)
        .def("get_type", &ATOMS_T::getType)
        .def("get_mass", &ATOMS_T::getMass)
        .def("get_charge", &ATOMS_T::getCharge)
        .def("get_relative_mass", &ATOMS_T::getRelativeMass)
        //        .def("get_position",
        //             [](const ATOMS_T &atoms)
        //             {
        //                 const auto &pos = atoms.getPos();
        //                 return py::array_t<double>({pos.extent(2), pos.extent(1), pos.extent(0)},
        //                                            {pos.stride(2), pos.stride(1), pos.stride(0)},
        //                                            pos.data());
        //             })
        .def_readwrite("num_local_atoms", &ATOMS_T::numLocalAtoms)
        .def_readwrite("num_ghost_atoms", &ATOMS_T::numGhostAtoms);
}

void init_data(py::module_ &m)
{
    using namespace mrmd;

    init_atoms<data::DeviceAtoms>(m, "DeviceAtoms");
    //    init_atoms<data::HostAtoms>(m, "HostAtoms");

    py::class_<data::MPIInfo>(m, "MPIInfo")
        .def(py::init<>())
        .def_readonly("rank", &data::MPIInfo::rank)
        .def_readonly("size", &data::MPIInfo::size);

    py::class_<data::Subdomain>(m, "Subdomain")
        .def(py::init<const std::array<real_t, 3> &, const std::array<real_t, 3> &, real_t>())
        .def_readonly("min_corner", &data::Subdomain::minCorner)
        .def_readonly("max_corner", &data::Subdomain::maxCorner)
        .def_readonly("ghost_layer_thickness", &data::Subdomain::ghostLayerThickness)
        .def_readonly("min_ghost_corner", &data::Subdomain::minGhostCorner)
        .def_readonly("max_ghost_corner", &data::Subdomain::maxGhostCorner)
        .def_readonly("min_inner_corner", &data::Subdomain::minInnerCorner)
        .def_readonly("max_inner_corner", &data::Subdomain::maxInnerCorner)
        .def_readonly("diameter", &data::Subdomain::diameter)
        .def_readonly("diameter_with_ghost_layer", &data::Subdomain::diameterWithGhostLayer);
}