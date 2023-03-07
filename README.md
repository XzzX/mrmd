![Build Status](https://img.shields.io/github/actions/workflow/status/xzzx/mrmd/validate.yml?branch=main&label=main&style=plastic)
![Lines of Code](https://img.shields.io/tokei/lines/github/xzzx/mrmd?label=lines%20of%20code&style=plastic)
![Code Size](https://img.shields.io/github/languages/code-size/xzzx/mrmd?style=plastic)

# MRMD

**M**ulti **R**esolution **M**olecular **D**ynamics

# How to Build

## Requirements

### Infrastructure

* A C++17 compatible compiler
* [CMake](https://cmake.org/) >= 3.19

### Libraries

* MPI
* [Kokkos](https://github.com/kokkos/kokkos)  
  You need to build it yourself for the intended target architecture and backend.
* parallel installation of [HDF5](https://www.hdfgroup.org/solutions/hdf5/)

Will be downloaded automatically and compiled alongside MRMD:

* [CLI11](https://github.com/CLIUtils/CLI11.git)
* [fmt](https://github.com/fmtlib/fmt.git)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp.git)
* [Cabana](https://github.com/ECP-copa/Cabana.git)
* [googletest](https://github.com/google/googletest.git)

## Build Instructions

```bash
git clone https://github.com/XzzX/mrmd
cmake -S mrmd -B mrmd-build -DCMAKE_BUILD_TYPE=Release
cmake --build mrmd-build --target all -j 8
```
