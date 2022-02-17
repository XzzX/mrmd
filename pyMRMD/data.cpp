#include "data.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <assert/assert.hpp>
#include <data/Atoms.hpp>
#include <data/MPIInfo.hpp>
#include <data/Molecules.hpp>
#include <data/MoleculesFromAtoms.hpp>
#include <data/Subdomain.hpp>

namespace py = pybind11;

template <class SLICE_T>
py::array_t<typename SLICE_T::value_type> getNPArray(const SLICE_T slice)
{
    py::str dummyDataOwner;  // https://github.com/pybind/pybind11/issues/323#issuecomment-575717041
    if (slice.rank() == 2)
    {
        return py::array_t<typename SLICE_T::value_type>(
            {slice.extent(1), slice.extent(0)},
            {sizeof(typename SLICE_T::value_type) * slice.stride(1),
             sizeof(typename SLICE_T::value_type) * slice.stride(0)},
            slice.data(),
            dummyDataOwner);
    }

    MRMD_HOST_CHECK_EQUAL(slice.rank(), 3);
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

    m.def("create_molecule_for_each_atom", &data::createMoleculeForEachAtom);
}