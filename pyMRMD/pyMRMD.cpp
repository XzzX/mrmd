#include <pybind11/pybind11.h>

#include <initialization.hpp>

#include "action.hpp"
#include "analysis.hpp"
#include "cabana.hpp"
#include "communication.hpp"
#include "data.hpp"
#include "io.hpp"
#include "util.hpp"

namespace py = pybind11;
PYBIND11_MODULE(pyMRMD, m)
{
    m.doc() = "MRMD Python Wrapper";

    m.def("initialize", py::overload_cast<>(&mrmd::initialize), "");
    m.def("finalize", &mrmd::finalize, "");

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
}