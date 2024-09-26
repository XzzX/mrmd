![Build Status](https://img.shields.io/github/actions/workflow/status/xzzx/mrmd/validate.yml?branch=main&label=main)
![GitHub License](https://img.shields.io/github/license/xzzx/mrmd)

# MRMD

**M**ulti **R**esolution **M**olecular **D**ynamics

# How to Build

## Requirements

### Infrastructure

* A C++17 compatible compiler
* [CMake](https://cmake.org/) >= 3.19

### Libraries

* MPI
* parallel installation of [HDF5](https://www.hdfgroup.org/solutions/hdf5/)

Will be downloaded automatically and compiled alongside MRMD:

* [CLI11](https://github.com/CLIUtils/CLI11.git)
* [fmt](https://github.com/fmtlib/fmt.git)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp.git)
* [Kokkos](https://github.com/kokkos/kokkos)
  You need to specify correct flags for the intended target architecture and backend.
* [Cabana](https://github.com/ECP-copa/Cabana.git)
* [googletest](https://github.com/google/googletest.git)

## CMake configuration options
### MRMD specific options
| Flag | Description | Default |
| ---- | ----------- | ------- |
| MRMD_CLANG_TIDY | Run clang-tidy checks during compilation. | OFF |
| MRMD_ENABLE_COVERAGE | Enable code coverage. (clang)" | OFF |
| MRMD_ENABLE_HDF5 | Enable HDF5 support. | OFF |
| MRMD_ENABLE_TESTING | Build tests and add them to ctest. | ON |
| MRMD_LOCAL_ARCHITECTURE | Use instruction set of the local architecture. | OFF |
| MRMD_VEC_REPORT | Enable reporting of loop vectorization. | OFF |
| MRMD_VERBOSE_ASSERTS | Verbose asserts are only available on CPU! | OFF |
| MRMD_WERROR | Treat warnings as errors. | OFF |

### Kokkos specific options
| Flag | Description | Options |
| ---- | ----------- | -------- |
| Kokkos_ENABLE_* | Enable Kokkos backends | SERIAL, OPENMP, CUDA, ... |
| Kokkos_ARCH_* | Select target architecture | AMPERE80, NATIVE, ... | 

## Build Instructions

```bash
git clone https://github.com/XzzX/mrmd
cmake -S mrmd -B mrmd-build -DCMAKE_BUILD_TYPE=Release -DMRMD_ENABLE_HDF5=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
cmake --build mrmd-build --target all -j 8
```
