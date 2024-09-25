#!python3
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


import numpy as np
import pyMRMD


def fill_domain_with_atoms_sc(config, subdomain):
    num_atoms_a = config["num_atoms"] * config["fraction_type_A"]

    atoms = pyMRMD.data.HostAtoms(config["num_atoms"])

    pos = atoms.get_pos_np()
    pos[0, :, :] = (
        np.random.uniform(size=pos.shape[1:]) * subdomain.diameter[0]
        + subdomain.min_corner[0]
    )
    pos[1, :, :] = (
        np.random.uniform(size=pos.shape[1:]) * subdomain.diameter[1]
        + subdomain.min_corner[1]
    )
    pos[2, :, :] = (
        np.random.uniform(size=pos.shape[1:]) * subdomain.diameter[2]
        + subdomain.min_corner[2]
    )

    vel = atoms.get_vel_np()
    vel[:, :, :] = np.random.uniform(-0.5, 0.5, size=vel.shape) * config["max_velocity"]

    mass = atoms.get_mass_np()
    mass[:, :] = 1

    relative_mass = atoms.get_relative_mass_np()
    relative_mass[:, :] = 1

    type = atoms.get_type_np()
    type[:, :] = 1
    type[:, : int(num_atoms_a / type.shape[0])] = 0

    atoms.num_local_atoms = config["num_atoms"]
    atoms.num_ghost_atoms = 0

    return pyMRMD.data.DeviceAtoms(atoms)


def init_molecules(num_atoms):
    molecules = pyMRMD.data.HostMolecules(2 * num_atoms)

    offset = molecules.get_atoms_offset_np()
    offset[:, :] = np.arange(offset.shape[0] * offset.shape[1]).reshape(offset.shape)

    size = molecules.get_num_atoms_np()
    size[:, :] = 1

    molecules.num_local_molecules = num_atoms

    return pyMRMD.data.DeviceMolecules(molecules)


def init_simulation(config):
    subdomain = pyMRMD.data.Subdomain(
        [0, 0, 0], config["box"], config["ghost_layer_thickness"]
    )
    atoms = fill_domain_with_atoms_sc(config, subdomain)
    return subdomain, atoms
