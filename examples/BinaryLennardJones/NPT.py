import numpy as np
import pyMRMD


def npt(config, subdomain, atoms):
    estimated_max_neighbors = 60
    cell_ratio = 0.5
    volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]
    rho = atoms.num_local_atoms / volume
    print("rho: ", rho)

    neighbor_cutoff = np.max(config["LJ"]["cutoff"]) + config["LJ"]["skin"]

    ghost_layer = pyMRMD.communication.GhostLayer()
    LJ = pyMRMD.action.LennardJones(
        config["LJ"]["capping"],
        config["LJ"]["cutoff"],
        config["LJ"]["sigma"],
        config["LJ"]["epsilon"],
        2,
        True,
    )
    verlet_list = pyMRMD.cabana.VerletList()
    timer = pyMRMD.util.Timer()

    maxAtomDisplacement = 1e10

    current_pressure = pyMRMD.util.ExponentialMovingAverage(
        config["pressure_averaging_coefficient"]
    )
    current_temperature = pyMRMD.util.ExponentialMovingAverage(
        config["temperature_averaging_coefficient"]
    )

    for step in range(config["time_steps"]):

        maxAtomDisplacement += pyMRMD.action.velocity_verlet.pre_force_integrate(
            atoms, config["dt"]
        )

        if (step > config["barostat_start"]) and (
            step % config["barostat_interval"] == 0
        ):
            pyMRMD.action.berendsen_barostat.apply(
                atoms,
                current_pressure.to_real(),
                config["target_pressure"],
                config["pressure_relaxation_coefficient"],
                subdomain,
                False,
                True,
                False,
            )
            volume = (
                subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]
            )
            maxAtomDisplacement = 1e10

        if step % config["thermostat_interval"] == 0:
            pyMRMD.action.berendsen_thermostat.apply(
                atoms,
                current_temperature.to_real(),
                config["target_temperature"],
                config["temperature_relaxation_coefficient"],
            )

        if maxAtomDisplacement >= config["LJ"]["skin"] * 0.5:

            # reset displacement
            maxAtomDisplacement = 0

            ghost_layer.exchange_real_atoms(atoms, subdomain)

            ghost_layer.create_ghost_atoms(atoms, subdomain)
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

        atoms.set_force(0)
        LJ.apply(atoms, verlet_list)

        if step < 201:
            current_pressure = pyMRMD.util.ExponentialMovingAverage(
                config["pressure_averaging_coefficient"]
            )
            current_temperature = pyMRMD.util.ExponentialMovingAverage(
                config["temperature_averaging_coefficient"]
            )

        Ek = pyMRMD.analysis.get_kinetic_energy(atoms)
        current_pressure.append(2 * (Ek - LJ.get_virial()) / (3 * volume))
        Ek /= atoms.num_local_atoms
        current_temperature.append((2.0 / 3.0) * Ek)

        ghost_layer.contribute_back_ghost_to_real(atoms)

        pyMRMD.action.velocity_verlet.post_force_integrate(atoms, config["dt"])

        if config["enable_output"] and (step % config["output_interval"] == 0):
            print(
                f"{step:>8} | {timer.seconds():>8.3} | {current_temperature.to_real():>8.3} | {current_pressure.to_real():>8.3} | {volume:>8.3} | {Ek:>8.3} | {LJ.get_energy() / atoms.num_local_atoms:>8.3} | {Ek + LJ.get_energy() / atoms.num_local_atoms:>8.3} | {atoms.num_local_atoms:>8} | {atoms.num_ghost_atoms:>8}"
            )
