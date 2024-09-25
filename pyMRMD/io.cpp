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