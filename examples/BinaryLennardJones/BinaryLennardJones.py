# Copyright 2024 Sebastian Eibl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pprint import pprint
import pyMRMD
from .initialization import init_simulation
from .NPT import npt
from .SPARTIAN import spartian
import yaml

config = yaml.load(open("input.yaml", "r", encoding="utf-8"), Loader=yaml.Loader)
pprint(config)

pyMRMD.initialize()
mpi_info = pyMRMD.data.MPIInfo()
subdomain = pyMRMD.data.Subdomain()
atoms = pyMRMD.data.DeviceAtoms(0)
dumper = pyMRMD.io.DumpH5MDParallel(mpi_info, "XzzX", "atoms")
restore = pyMRMD.io.RestoreH5MDParallel(mpi_info, "atoms")
# restore.restore('nvt.hdf5', subdomain, atoms)
# print(subdomain.min_corner, subdomain.max_corner)

subdomain, atoms = init_simulation(config["initialization"])
dumper.dump("initial.hdf5", subdomain, atoms)

npt(config["NPT"], subdomain, atoms)
dumper.dump("npt.hdf5", subdomain, atoms)

npt(config["NVT"], subdomain, atoms)
dumper.dump("nvt.hdf5", subdomain, atoms)

molecules = pyMRMD.data.create_molecule_for_each_atom(atoms)
spartian(config["SPARTIAN"], subdomain, atoms, molecules)
dumper.dump("spartian.hdf5", subdomain, atoms)
