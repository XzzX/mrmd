#include "weighting_function.hpp"

#include <pybind11/stl.h>

#include <datatypes.hpp>
#include <weighting_function/Slab.hpp>

namespace py = pybind11;

void init_weighting_function(py::module_ &m)
{
    using namespace mrmd;
    py::class_<weighting_function::Slab>(m, "Slab")
        .def(py::init<std::array<real_t, 3> &, real_t, real_t, idx_t>())
        .def("is_in_at_region", &weighting_function::Slab::isInATRegion)
        .def("is_in_hy_region", &weighting_function::Slab::isInHYRegion)
        .def("is_in_cg_region", &weighting_function::Slab::isInCGRegion);
}