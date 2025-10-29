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


def spartian(config, subdomain, atoms, molecules):
    estimated_max_neighbors = 60
    cell_ratio = 0.5
    volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]
    rho = atoms.num_local_atoms / volume
    print("rho: ", rho)

    neighbor_cutoff = np.max(config["LJ"]["cutoff"]) + config["LJ"]["skin"]

    ghost_layer = pyMRMD.communication.MultiResGhostLayer()
    LJ = pyMRMD.action.LJ_IdealGas(
        config["LJ"]["capping"],
        config["LJ"]["cutoff"],
        config["LJ"]["sigma"],
        config["LJ"]["epsilon"],
        2,
        True,
    )
    LJ.set_compensation_energy_sampling_interval(
        config["compensation_energy_sampling_interval"]
    )
    LJ.set_compensation_energy_sampling_interval(
        config["compensation_energy_update_interval"]
    )
    verlet_list = pyMRMD.cabana.VerletList()
    thermostat = pyMRMD.action.LangevinThermostat(
        config["temperature_relaxation_coefficient"],
        config["target_temperature"],
        config["dt"],
    )
    timer = pyMRMD.util.Timer()

    maxAtomDisplacement = 1e10

    current_pressure = pyMRMD.util.ExponentialMovingAverage(
        config["pressure_averaging_coefficient"]
    )
    current_temperature = pyMRMD.util.ExponentialMovingAverage(
        config["temperature_averaging_coefficient"]
    )

    weighting_function = pyMRMD.weighting_function.Slab(
        config["center"],
        config["atomistic_region_diameter"],
        config["hybrid_region_diameter"],
        config["lambda_exponent"],
    )

    h_atoms = pyMRMD.data.HostAtoms(atoms)
    rhoA = (h_atoms.get_type_np() == 0).sum() / volume
    rhoB = (h_atoms.get_type_np() == 1).sum() / volume

    thermodynamic_force = pyMRMD.action.ThermodynamicForce(
        [rhoA, rhoB],
        subdomain,
        config["density_bin_width"],
        config["thermodynamic_force_modulation"],
        config["thermodynamic_force_use_symmetry"],
        config["thermodynamic_force_use_periodicity"],
    )

    Xrho1 = 0.0
    Xrho2 = 0.0
    for step in range(config["time_steps"]):

        maxAtomDisplacement += pyMRMD.action.velocity_verlet.pre_force_integrate(
            atoms, config["dt"]
        )

        pyMRMD.action.update_molecules.update(molecules, atoms, weighting_function)

        if maxAtomDisplacement >= config["LJ"]["skin"] * 0.5:

            # reset displacement
            maxAtomDisplacement = 0

            ghost_layer.exchange_real_atoms(molecules, atoms, subdomain)

            ghost_layer.create_ghost_atoms(molecules, atoms, subdomain)
            pyMRMD.cabana.build_verlet_list(
                verlet_list,
                atoms,
                subdomain,
                neighbor_cutoff,
                cell_ratio,
                estimated_max_neighbors,
            )
        else:
            ghost_layer.update_ghost_atoms(atoms, subdomain)

        pyMRMD.action.update_molecules.update(molecules, atoms, weighting_function)

        atoms.set_force(0)
        molecules.set_force(0)

        if step % config["thermostat_interval"] == 0:
            thermostat.apply(atoms)

        if (step > config["density_start"]) and (
            step % config["density_sampling_interval"] == 0
        ):
            thermodynamic_force.sample(atoms)

        if (step > config["density_start"]) and (
            step % config["density_update_interval"] == 0
        ):
            density_profile = pyMRMD.analysis.get_axial_density_profile(
                atoms.num_local_atoms,
                atoms,
                2,
                thermodynamic_force.get_density_profile().min,
                thermodynamic_force.get_density_profile().max,
                thermodynamic_force.get_density_profile().numBins,
                pyMRMD.AXIS_X,
            )
            density_profile.scale(
                1
                / (
                    density_profile.binSize
                    * subdomain.diameter[1]
                    * subdomain.diameter[2]
                )
            )
            Xrho1 = pyMRMD.analysis.get_fluctuation(density_profile, rhoA, 0)
            Xrho2 = pyMRMD.analysis.get_fluctuation(density_profile, rhoB, 1)

            thermodynamic_force.update(
                config["smoothing_sigma"], config["smoothing_intensity"]
            )

        if step > config["density_start"]:
            thermodynamic_force.apply(atoms, weighting_function)

        LJ.run(molecules, verlet_list, atoms)
        pyMRMD.action.contribute_molecule_force_to_atoms.update(molecules, atoms)

        if step < 201:
            current_pressure = pyMRMD.util.ExponentialMovingAverage(
                config["pressure_averaging_coefficient"]
            )
            current_temperature = pyMRMD.util.ExponentialMovingAverage(
                config["temperature_averaging_coefficient"]
            )

        Ek = pyMRMD.analysis.get_kinetic_energy(atoms)
        # current_pressure.append(2 * (Ek - LJ.get_virial()) / (3 * volume))
        Ek /= atoms.num_local_atoms
        current_temperature.append((2.0 / 3.0) * Ek)

        ghost_layer.contribute_back_ghost_to_real(atoms)

        pyMRMD.action.velocity_verlet.post_force_integrate(atoms, config["dt"])

        if config["enable_output"] and (step % config["output_interval"] == 0):
            mu_left = thermodynamic_force.get_mu_left()
            mu_right = thermodynamic_force.get_mu_right()
            print(
                f"{step:>8} | {timer.seconds():>8.3} | {current_temperature.to_real():>8.3} | {current_pressure.to_real():>8.3} | {volume:>8.3} | {mu_left[1]:>8.3} | {mu_right[1]:>8.3} | {Xrho1:>8.3} | {Xrho2:>8} | {atoms.num_ghost_atoms:>8}"
            )
