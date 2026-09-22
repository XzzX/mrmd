![Build Status](https://img.shields.io/github/actions/workflow/status/xzzx/mrmd/validate.yml?branch=main&label=main)
![GitHub License](https://img.shields.io/github/license/xzzx/mrmd)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17698862.svg)](https://doi.org/10.5281/zenodo.17698862)

# MRMD - **M**ulti **R**esolution **M**olecular **D**ynamics

MRMD is a stand-alone, open-source C++ package implementing the (Hamiltonian) adaptive resolution
simulation ((H-)AdResS) method, which concurrently couples regions of different resolution — e.g.
atomistic Lennard-Jones and ideal gas — to simulate open molecular systems that exchange particles
and energy with a reservoir.

The software exposes all of its algorithms as composable building blocks,
facilitating both running AdResS simulations and developing the
method itself.

# How to Build

## Requirements

### Infrastructure

- A C++ compiler with C++20 support
- [CMake](https://cmake.org/) >= 3.25

### Libraries

#### Integrated libraries

These are fetched and built automatically by CMake:

- [CLI11](https://github.com/CLIUtils/CLI11.git)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp.git)
- [Kokkos](https://github.com/kokkos/kokkos) — specify flags for your target architecture and backend
- [Cabana](https://github.com/ECP-copa/Cabana.git)
- [googletest](https://github.com/google/googletest.git)

#### Optional libraries

- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) — required for H5MD I/O support

## CMake configuration options

### MRMD specific options

| Flag                  | Description                                | Default |
| --------------------- | ------------------------------------------ | ------- |
| MRMD_ENABLE_COVERAGE  | Enable code coverage (clang only)          | OFF     |
| MRMD_ENABLE_HDF5      | Enable HDF5 / H5MD support                 | OFF     |
| MRMD_ENABLE_TESTING   | Build tests and add them to ctest          | ON      |
| MRMD_USE_SHARED_SPACE | Use shared space for Kokkos.               | OFF     |
| MRMD_VEC_REPORT       | Enable reporting of loop vectorization     | OFF     |
| MRMD_VERBOSE_ASSERTS  | Verbose asserts (CPU only)                 | OFF     |
| MRMD_WERROR           | Treat warnings as errors                   | OFF     |

### Kokkos specific options

| Flag              | Description                | Options                   |
| ----------------- | -------------------------- | ------------------------- |
| Kokkos_ENABLE_\*  | Enable Kokkos backends     | SERIAL, OPENMP, CUDA, ... |
| Kokkos_ARCH_\*    | Select target architecture | AMPERE80, NATIVE, ...     |

## Build Instructions

```bash
git clone https://github.com/XzzX/mrmd
cmake -S mrmd \
      -B mrmd-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ARCH_NATIVE=ON
cmake --build mrmd-build --parallel 8
cd mrmd-build/examples/00_IdealGas_NVE
./00_IdealGas_NVE
```

## Running the Tests

```bash
cd mrmd-build
ctest --parallel 4 --output-on-failure
```

# Tutorial

The [`examples`](examples) directory contains a series of self-contained simulations that build on each other,
from a minimal, non-interacting system up to a full multi-resolution production workflow. Each example is built
as its own executable in `mrmd-build/examples/<name>`. Work through them in order to get familiar with MRMD.

| Example                                                                            | Description                                                                                                                                                             |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`00_IdealGas_NVE`](examples/00_IdealGas_NVE)                                       | Minimal ideal gas (non-interacting particles) simulation in the NVE (microcanonical) ensemble. Introduces the simulation domain, ghost layer, and the velocity-Verlet integrator. |
| [`01_IdealGas_NVT`](examples/01_IdealGas_NVT)                                       | Same ideal gas setup as above, but run in the NVT (canonical) ensemble using a Langevin thermostat.                                                                     |
| [`02_LennardJones_NVT`](examples/02_LennardJones_NVT)                               | Lennard-Jones fluid in the NVT ensemble with a Langevin thermostat, neighbor lists, and analysis of kinetic energy, pressure, and system momentum. Writes a `.gro` configuration file used by later examples. |
| [`03_LennardJones_NVE`](examples/03_LennardJones_NVE)                               | Lennard-Jones fluid in the NVE ensemble, restoring the initial configuration from the `.gro` file produced by `02_LennardJones_NVT` and writing H5MD output.            |
| [`04_LennardJones_IdealGas_LocalCap`](examples/04_LennardJones_IdealGas_LocalCap)   | Relaxes a randomly placed, ideal-gas-like starting configuration into a Lennard-Jones fluid using a locally capped force during equilibration to avoid instabilities from overlapping atoms. |
| [`05_LennardJones_Equilibration`](examples/05_LennardJones_Equilibration)           | Two-stage equilibration workflow: first with a Berendsen thermostat, then a follow-up run with a Langevin thermostat, producing H5MD checkpoints for production runs.   |
| [`06_LennardJones_ProductionAtomistic`](examples/06_LennardJones_ProductionAtomistic) | Atomistic production simulation restoring from an equilibrated H5MD checkpoint, with configurable thermostat and force-capping options.                                 |
| [`07_LennardJones_ThermoForce`](examples/07_LennardJones_ThermoForce)               | Computes the thermodynamic force field iteratively for adaptive-resolution coupling; includes a variant that starts from a precomputed initial guess to speed up convergence. |
| [`08_LennardJones_ProductionTracer`](examples/08_LennardJones_ProductionTracer)     | Production run with tracer particles using the converged thermodynamic force field from `07_LennardJones_ThermoForce`.                                                  |

# Contributing

Contributions are welcome! Please open an issue to discuss a bug or feature request before submitting a pull request. When contributing code, follow the existing code style (enforced via `.clang-format`) and ensure all tests pass.

# Citation

If you use MRMD in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff) or the following:

> Sebastian Eibl and Julian Friedrich Hille. *Multi Resolution Molecular Dynamics (MRMD)*. doi:[10.5281/zenodo.17698862](https://doi.org/10.5281/zenodo.17698862)

# Authors

- Sebastian Eibl ([@XzzX](https://github.com/XzzX)) — [ORCID](https://orcid.org/0000-0002-1069-2720)
- Julian Friedrich Hille ([@J-Hizzle](https://github.com/J-Hizzle)) — [ORCID](https://orcid.org/0009-0008-1005-9053)

# License

MRMD is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
