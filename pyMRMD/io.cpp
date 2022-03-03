#include "io.hpp"

#include <io/DumpCSV.hpp>
#include <io/DumpGRO.hpp>
#include <io/DumpH5MDParallel.hpp>
#include <io/RestoreH5MDParallel.hpp>

namespace py = pybind11;

void init_io(py::module_ &m)
{
    using namespace mrmd;
    m.def("dump_csv", &io::dumpCSV);
    m.def("dump_gro", &io::dumpGRO);

    py::class_<io::DumpH5MDParallel>(m, "DumpH5MDParallel")
        .def(py::init<const std::shared_ptr<data::MPIInfo> &,
                      const std::string &,
                      const std::string &>())
        .def("dump", &io::DumpH5MDParallel::dump);

    py::class_<io::RestoreH5MDParallel>(m, "RestoreH5MDParallel")
        .def(py::init<const std::shared_ptr<data::MPIInfo> &, const std::string &>())
        .def("restore", &io::RestoreH5MDParallel::restore);
}