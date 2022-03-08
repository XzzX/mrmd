#include <pybind11/pybind11.h>

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