import numpy as np
from pprint import pprint
import pyMRMD
from initialization import init_simulation, init_molecules
from NPT import npt
from SPARTIAN import spartian
import yaml

config = yaml.load(open("input.yaml", "r"), Loader=yaml.Loader)
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
