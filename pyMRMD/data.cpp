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

#include "data.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <assert/assert.hpp>
#include <data/Atoms.hpp>
#include <data/MPIInfo.hpp>
#include <data/Molecules.hpp>
#include <data/MoleculesFromAtoms.hpp>
#include <data/MultiHistogram.hpp>
#include <data/Subdomain.hpp>

namespace py = pybind11;

template <class SLICE_T>
py::array_t<typename SLICE_T::value_type> getNPArray(const SLICE_T& slice)
{
    static_assert(slice.viewRank() == 2 || slice.viewRank() == 3, "Invalid view rank dimension");

    py::str dummyDataOwner;  // https://github.com/pybind/pybind11/issues/323#issuecomment-575717041
    if constexpr(slice.viewRank() == 2)
    {
        return py::array_t<typename SLICE_T::value_type>(
            {slice.extent(1), slice.extent(0)},
            {sizeof(typename SLICE_T::value_type) * slice.stride(1),
             sizeof(typename SLICE_T::value_type) * slice.stride(0)},
            slice.data(),
            dummyDataOwner);
    }

    if constexpr(slice.viewRank() == 3)
        return py::array_t<typename SLICE_T::value_type>(
            {slice.extent(2), slice.extent(1), slice.extent(0)},
            {sizeof(typename SLICE_T::value_type) * slice.stride(2),
            sizeof(typename SLICE_T::value_type) * slice.stride(1),
            sizeof(typename SLICE_T::value_type) * slice.stride(0)},
            slice.data(),
            dummyDataOwner);
}

template <class ATOMS_T>
void init_atoms(py::module_ &m, const char *className)
{
    using namespace mrmd;

    py::class_<ATOMS_T>(m, className)
        .def(py::init<const idx_t>())
        .def(py::init<const data::DeviceAtoms &>())
        .def(py::init<const data::HostAtoms &>())
        .def("get_pos", &ATOMS_T::getPos)
        .def("get_vel", &ATOMS_T::getVel)
        .def("get_force", &ATOMS_T::getForce)
        .def("get_type", &ATOMS_T::getType)
        .def("get_mass", &ATOMS_T::getMass)
        .def("get_charge", &ATOMS_T::getCharge)
        .def("get_relative_mass", &ATOMS_T::getRelativeMass)
        .def("get_pos_np", [](const ATOMS_T &atoms) { return getNPArray(atoms.getPos()); })
        .def("get_vel_np", [](const ATOMS_T &atoms) { return getNPArray(atoms.getVel()); })
        .def("get_force_np", [](const ATOMS_T &atoms) { return getNPArray(atoms.getForce()); })
        .def("get_type_np", [](const ATOMS_T &atoms) { return getNPArray(atoms.getType()); })
        .def("get_mass_np", [](const ATOMS_T &atoms) { return getNPArray(atoms.getMass()); })
        .def("get_charge_np", [](const ATOMS_T &atoms) { return getNPArray(atoms.getCharge()); })
        .def("get_relative_mass_np",
             [](const ATOMS_T &atoms) { return getNPArray(atoms.getRelativeMass()); })
        .def("set_force", &ATOMS_T::setForce)
        .def_readwrite("num_local_atoms", &ATOMS_T::numLocalAtoms)
        .def_readwrite("num_ghost_atoms", &ATOMS_T::numGhostAtoms);
}

template <class MOLECULES_T>
void init_molecules(py::module_ &m, const char *className)
{
    using namespace mrmd;

    py::class_<MOLECULES_T>(m, className)
        .def(py::init<const idx_t>())
        .def(py::init<const data::DeviceMolecules &>())
        .def(py::init<const data::HostMolecules &>())
        .def("get_pos", &MOLECULES_T::getPos)
        .def("get_force", &MOLECULES_T::getForce)
        .def("get_pos_np",
             [](const MOLECULES_T &molecules) { return getNPArray(molecules.getPos()); })
        .def("get_force_np",
             [](const MOLECULES_T &molecules) { return getNPArray(molecules.getForce()); })
        .def("set_force", &MOLECULES_T::setForce)
        .def_readwrite("num_local_atoms", &MOLECULES_T::numLocalMolecules)
        .def_readwrite("num_ghost_atoms", &MOLECULES_T::numGhostMolecules);
}

void init_data(py::module_ &m)
{
    using namespace mrmd;

    init_atoms<data::DeviceAtoms>(m, "DeviceAtoms");
    init_atoms<data::HostAtoms>(m, "HostAtoms");

    init_molecules<data::DeviceMolecules>(m, "DeviceMolecules");
    init_molecules<data::HostMolecules>(m, "HostMolecules");

    py::class_<data::MPIInfo, std::shared_ptr<data::MPIInfo>>(m, "MPIInfo")
        .def(py::init<>())
        .def_readonly("rank", &data::MPIInfo::rank)
        .def_readonly("size", &data::MPIInfo::size);

    py::class_<data::MultiHistogram>(m, "MultiHistogram")
        .def(py::init<const std::string &, const real_t, const real_t, idx_t, idx_t>())
        .def("scale", py::overload_cast<const real_t &>(&data::MultiHistogram::scale))
        .def("makeSymmetric", &data::MultiHistogram::makeSymmetric)
        .def_readonly("min", &data::MultiHistogram::min)
        .def_readonly("max", &data::MultiHistogram::max)
        .def_readonly("numBins", &data::MultiHistogram::numBins)
        .def_readonly("numHistograms", &data::MultiHistogram::numHistograms)
        .def_readonly("binSize", &data::MultiHistogram::binSize)
        .def_readonly("inverseBinSize", &data::MultiHistogram::inverseBinSize);
    
    m.def("cumulativeMovingAverage", &data::cumulativeMovingAverage);
    m.def("gradient", &data::gradient);
    m.def("smoothen", &data::smoothen);

    py::class_<data::Subdomain>(m, "Subdomain")
        .def(py::init<>())
        .def(py::init<const Point3D &, const Point3D &, real_t>())
        .def_readonly("min_corner", &data::Subdomain::minCorner)
        .def_readonly("max_corner", &data::Subdomain::maxCorner)
        .def_readonly("ghost_layer_thickness", &data::Subdomain::ghostLayerThickness)
        .def_readonly("min_ghost_corner", &data::Subdomain::minGhostCorner)
        .def_readonly("max_ghost_corner", &data::Subdomain::maxGhostCorner)
        .def_readonly("min_inner_corner", &data::Subdomain::minInnerCorner)
        .def_readonly("max_inner_corner", &data::Subdomain::maxInnerCorner)
        .def_readonly("diameter", &data::Subdomain::diameter)
        .def_readonly("diameter_with_ghost_layer", &data::Subdomain::diameterWithGhostLayer);

    m.def("create_molecule_for_each_atom", &data::createMoleculeForEachAtom);
}
