#include "io.hpp"

#include <io/DumpCSV.hpp>
#include <io/DumpGRO.hpp>

namespace py = pybind11;

void init_io(py::module_ &m)
{
    m.def("dump_csv", &mrmd::io::dumpCSV);
    m.def("dump_gro", &mrmd::io::dumpGRO);
}