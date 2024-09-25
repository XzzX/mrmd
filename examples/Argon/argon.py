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


config = Config()


def fill_domain_with_atoms_sc(subdomain, num_atoms, max_velocity):
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
    atoms.get_type_np()[:, :] = 0

    atoms.num_local_atoms = num_atoms
    atoms.num_ghost_atoms = 0

    return pyMRMD.data.DeviceAtoms(atoms)


pyMRMD.initialize()
subdomain = pyMRMD.data.Subdomain(
    [0, 0, 0], [config.Lx, config.Lx, config.Lx], config.neighbor_cutoff
)
volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]
atoms = fill_domain_with_atoms_sc(subdomain, config.num_atoms, 1)
rho = atoms.num_local_atoms / volume
print("rho: ", rho)

ghost_layer = pyMRMD.communication.GhostLayer()
LJ = pyMRMD.action.LennardJones(
    config.rc, config.sigma, config.epsilon, 0.7 * config.sigma
)
langevin_thermostat = pyMRMD.action.LangevinThermostat(
    config.gamma, config.temperature, config.dt
)
mean_square_displacement = pyMRMD.analysis.MeanSquareDisplacement()
mean_square_displacement.reset(atoms)
verlet_list = pyMRMD.cabana.VerletList()
timer = pyMRMD.util.Timer()

maxAtomDisplacement = 1e10
rebuildCounter = 0
msd = 0.0

with open("statistics.txt", "w") as fStat:
    for step in range(config.nsteps):

        maxAtomDisplacement += pyMRMD.action.velocity_verlet.pre_force_integrate(
            atoms, config.dt
        )

        if maxAtomDisplacement >= config.skin * 0.5:

            # reset displacement
            maxAtomDisplacement = 0

            ghost_layer.exchange_real_atoms(atoms, subdomain)

            ghost_layer.create_ghost_atoms(atoms, subdomain)
            pyMRMD.cabana.build_verlet_list(
                verlet_list,
                atoms,
                subdomain,
                config.neighbor_cutoff,
                config.cell_ratio,
                config.estimated_max_neighbors,
            )
            ++rebuildCounter
        else:
            ghost_layer.update_ghost_atoms(atoms, subdomain)

        atoms.set_force(0)
        LJ.apply(atoms, verlet_list)

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
            msd = mean_square_displacement.calc(atoms, subdomain) / (1000 * config.dt)
            if (config.temperature > 0) and (step > 5000):
                config.temperature -= 7.8e-3
                if config.temperature < 0:
                    config.temperature = 0

            langevin_thermostat.set(config.gamma, config.temperature * 0.5, config.dt)
            langevin_thermostat.apply(atoms)

            mean_square_displacement.reset(atoms)

        pyMRMD.action.velocity_verlet.post_force_integrate(atoms, config.dt)
