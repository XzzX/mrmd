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

import argparse
import numpy as np
import pyMRMD


class Config:
    def __init__(self):
        self.bOutput = True
        self.outputInterval = 10000

        self.nsteps = 400001
        self.num_atoms = 16 * 16 * 16

        self.sigma = 0.34  # units: nm
        self.epsilon = 0.993653  # units: kJ/mol
        self.mass = 39.948  # units: u

        self.rc = 2.3 * self.sigma
        self.skin = 0.1
        self.neighbor_cutoff = self.rc + self.skin

        self.dt = 0.00217  # units: ps
        self.temperature = 3.0
        self.gamma = 0.7 / self.dt

        self.Lx = 3.196 * 2  # units: nm

        self.cell_ratio = 1.0

        self.estimated_max_neighbors = 60


def fill_domain_with_atoms_sc(config, subdomain, num_atoms, max_velocity):
    atoms = pyMRMD.data.HostAtoms(num_atoms)

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
    vel[:, :, :] = np.random.uniform(-0.5, 0.5, size=vel.shape) * max_velocity

    mass = atoms.get_mass_np()
    mass[:, :] = config.mass

    atoms.get_type_np()[:, :] = 0
    atoms.get_charge_np()[:, :] = 0
    atoms.get_relative_mass_np()[:, :] = 1

    atoms.num_local_atoms = num_atoms
    atoms.num_ghost_atoms = 0

    return pyMRMD.data.DeviceAtoms(atoms)


def get_config():
    config = Config()

    parser = argparse.ArgumentParser(
        prog="Argon", description="Simulating the cooling of argon."
    )
    parser.add_argument(
        "-n",
        "--nsteps",
        type=int,
        default=config.nsteps,
        help="number of simulation steps",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=int,
        default=config.outputInterval,
        help="output interval",
    )
    args = parser.parse_args()

    config.nsteps = args.nsteps
    config.outputInterval = args.output

    return config


def main():
    config = get_config()
    pyMRMD.initialize()
    subdomain = pyMRMD.data.Subdomain(
        [0, 0, 0], [config.Lx, config.Lx, config.Lx], config.neighbor_cutoff
    )
    volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]
    atoms = fill_domain_with_atoms_sc(config, subdomain, config.num_atoms, 1)
    molecules = pyMRMD.data.create_molecule_for_each_atom(atoms)
    rho = atoms.num_local_atoms / volume
    print("rho: ", rho)

    ghost_layer = pyMRMD.communication.MultiResGhostLayer()
    weighting_function = pyMRMD.weighting_function.Slab([-100, -100, -100], 1, 1, 1)
    LJ = pyMRMD.action.LennardJones(
        config.rc, config.sigma, config.epsilon, 0.7 * config.sigma
    )
    langevin_thermostat = pyMRMD.action.LangevinThermostat(
        config.gamma, config.temperature, config.dt
    )
    mean_square_displacement = pyMRMD.analysis.MeanSquareDisplacement()
    mean_square_displacement.reset(atoms)
    verlet_list = pyMRMD.cabana.HalfVerletList()
    timer = pyMRMD.util.Timer()

    maxAtomDisplacement = 1e10
    rebuildCounter = 0
    msd = 0.0

    with open("statistics.txt", "w", encoding="utf-8") as fStat:
        for step in range(config.nsteps):

            maxAtomDisplacement += pyMRMD.action.velocity_verlet.pre_force_integrate(
                atoms, config.dt
            )

            pyMRMD.action.update_molecules.update(molecules, atoms, weighting_function)

            if maxAtomDisplacement >= config.skin * 0.5:

                # reset displacement
                maxAtomDisplacement = 0

                ghost_layer.exchange_real_atoms(molecules, atoms, subdomain)

                ghost_layer.create_ghost_atoms(molecules, atoms, subdomain)
                pyMRMD.cabana.build_verlet_list(
                    verlet_list,
                    atoms,
                    subdomain,
                    config.neighbor_cutoff,
                    config.cell_ratio,
                    config.estimated_max_neighbors,
                )
                rebuildCounter += 1
            else:
                ghost_layer.update_ghost_atoms(atoms, subdomain)

            pyMRMD.action.update_molecules.update(molecules, atoms, weighting_function)

            atoms.set_force(0)
            molecules.set_force(0)

            LJ.apply(atoms, verlet_list)
            pyMRMD.action.contribute_molecule_force_to_atoms.update(molecules, atoms)

            ghost_layer.contribute_back_ghost_to_real(atoms)

            if config.bOutput and (step % config.outputInterval == 0):
                E0 = LJ.get_energy() / atoms.num_local_atoms
                Ek = pyMRMD.analysis.get_mean_kinetic_energy(atoms)
                T = (2 / 3) * Ek
                p = pyMRMD.analysis.get_pressure(atoms, subdomain)
                print(
                    f"{step:>8} | {timer.seconds():>8.3} | {T:>8.3} | {Ek:>8.3} | {E0:>8.3} | {E0 + Ek:>8.3} | {p:>8.3} | {msd:>8.3} | {atoms.num_local_atoms:>8} | {atoms.num_ghost_atoms:>8}"
                )

                fStat.write(
                    f"{step:>8} {timer.seconds():>8} {T:>8} {Ek:>8} {E0:>8} {E0 + Ek:>8} {p:>8} {msd:>8} {atoms.num_local_atoms:>8} {atoms.num_ghost_atoms:>8}\n"
                )

            #            pyMRMD.io.dump_gro(f'argon_{step:0>6}.gro',
            #                               atoms,
            #                               subdomain,
            #                               step * config.dt,
            #                               "Argon",
            #                               False,
            #                               False)

            if step % 1000 == 0:
                msd = mean_square_displacement.calc(atoms, subdomain) / (
                    1000 * config.dt
                )
                if (config.temperature > 0) and (step > 5000):
                    config.temperature -= 7.8e-3
                    if config.temperature < 0:
                        config.temperature = 0

                langevin_thermostat.set(
                    config.gamma, config.temperature * 0.5, config.dt
                )
                langevin_thermostat.apply(atoms)

                mean_square_displacement.reset(atoms)

            pyMRMD.action.velocity_verlet.post_force_integrate(atoms, config.dt)


if __name__ == "__main__":
    main()
