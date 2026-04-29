![Build Status](https://img.shields.io/github/actions/workflow/status/xzzx/mrmd/validate.yml?branch=main&label=main)
![GitHub License](https://img.shields.io/github/license/xzzx/mrmd)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17698862.svg)](https://doi.org/10.5281/zenodo.17698862)

# MRMD

**M**ulti **R**esolution **M**olecular **D**ynamics

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

| Flag                 | Description                                | Default |
| -------------------- | ------------------------------------------ | ------- |
| MRMD_ENABLE_COVERAGE | Enable code coverage (clang only)          | OFF     |
| MRMD_ENABLE_HDF5     | Enable HDF5 / H5MD support                 | OFF     |
| MRMD_ENABLE_TESTING  | Build tests and add them to ctest          | ON      |
| MRMD_VEC_REPORT      | Enable reporting of loop vectorization     | OFF     |
| MRMD_VERBOSE_ASSERTS | Verbose asserts (CPU only)                 | OFF     |
| MRMD_WERROR          | Treat warnings as errors                   | OFF     |

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
cd mrmd-build/examples/Argon
./Argon
```

## Running the Tests

```bash
cd mrmd-build
ctest --parallel 4 --output-on-failure
```

# Contributing

Contributions are welcome! Please open an issue to discuss a bug or feature request before submitting a pull request. When contributing code, follow the existing code style (enforced via `.clang-format`) and ensure all tests pass.

# Citation

If you use MRMD in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff) or the following:

> Sebastian Eibl and Julian Friedrich Hille. *Multi Resolution Molecular Dynamics (MRMD)*. doi:[10.5281/zenodo.17698862](https://doi.org/10.5281/zenodo.17698862)

# Authors

- Sebastian Eibl ([@XzzX](https://github.com/XzzX)) — [ORCID](https://orcid.org/0000-0002-1069-2720)
- Julian Friedrich Hille — [ORCID](https://orcid.org/0009-0008-1005-9053)

# License

MRMD is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
