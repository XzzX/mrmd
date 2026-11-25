![Build Status](https://img.shields.io/github/actions/workflow/status/xzzx/mrmd/validate.yml?branch=main&label=main)
![GitHub License](https://img.shields.io/github/license/xzzx/mrmd)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue)](https://xzzx.github.io/mrmd/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17698862.svg)](https://doi.org/10.5281/zenodo.17698862)

# MRMD - Multi Resolution Molecular Dynamics

MRMD is a stand-alone, open-source C++ package implementing the (Hamiltonian) adaptive resolution
simulation ((H-)AdResS) method, which concurrently couples regions of different resolution — e.g.
atomistic Lennard-Jones and ideal gas — to simulate open molecular systems that exchange particles
and energy with a reservoir.

The software exposes all of its algorithms as composable building blocks,
facilitating both running AdResS simulations and developing the
method itself.

# Quickstart

For exhaustive installation instructions see the [documentation](https://xzzx.github.io/mrmd/getting_started.html).

## Requirements

- A C++ compiler with C++20 support
- [CMake](https://cmake.org/) >= 3.25

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

## Installing the Library

MRMD can be installed and consumed as a library from other CMake projects:

```bash
cmake --install mrmd-build --prefix /path/to/install
```

## Building the Documentation

The API documentation is published at <https://xzzx.github.io/mrmd/>. To build it
locally, install [Doxygen](https://www.doxygen.nl/) (and optionally
[Graphviz](https://graphviz.org/) for diagrams) and run:

```bash
cmake -S mrmd -B mrmd-build -DMRMD_BUILD_DOCS=ON
cmake --build mrmd-build --target docs
```

The HTML output is written to `mrmd-build/docs/html/index.html`.

# Tutorial

The [`examples`](examples) directory contains a series of self-contained simulations that build on each other,
from a minimal, non-interacting system up to a full multi-resolution production workflow. Each example is built
as its own executable in `mrmd-build/examples/<name>`. Work through them in numerical order to get familiar with MRMD.
See the [tutorials section](https://xzzx.github.io/mrmd/tutorials.html) in the documentation for further details. 

# Contributing

Contributions are welcome! Please open an issue to discuss a bug or feature request before submitting a pull request. When contributing code, follow the existing code style (enforced via `.clang-format`) and ensure all tests pass.

## Quality-of-life tooling: automatic formatting

[`format.sh`](format.sh) formats all tracked C++ sources (`*.cpp`, `*.hpp`) in place according to
[`.clang-format`](.clang-format). It is also wired up as a [pre-commit](https://pre-commit.com/) hook in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml), so formatting happens automatically on every commit.

Run it manually from the repository root:

```bash
./format.sh
```

Enable the commit hook once per clone:

```bash
pip install pre-commit
pre-commit install
```

Hard dependencies of the script:

- `clang-format-18` — exactly this versioned binary name must be on your `PATH`; a plain `clang-format` or a
  different version will not be picked up

The pre-commit hook additionally requires [`pre-commit`](https://pre-commit.com/) itself.

# Citation

If you use MRMD in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff) or the following:

> Sebastian Eibl and Julian Friedrich Hille. *Multi Resolution Molecular Dynamics (MRMD)*. doi:[10.5281/zenodo.17698862](https://doi.org/10.5281/zenodo.17698862)

# Authors

- Sebastian Eibl ([\@XzzX](https://github.com/XzzX)) — [ORCID](https://orcid.org/0000-0002-1069-2720)
- Julian Friedrich Hille ([\@J-Hizzle](https://github.com/J-Hizzle)) — [ORCID](https://orcid.org/0009-0008-1005-9053)

# License

MRMD is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
