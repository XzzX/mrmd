---
title: 'MRMD: A C++ package for Multi-Resolution Molecular Dynamics'
tags:
  - C++
  - molecular dynamics
  - adaptive resolution simulation
  - open molecular systems

authors:
  - name: Sebastian Eibl
    orcid: 0000-0002-1069-2720
    affiliation: 1
  - name: Julian F. Hille
    orcid: 0009-0008-1005-9053
    affiliation: 2
affiliations:
 - name: Max Planck Computing and Data Facility, Germany
   index: 1
   ror: 03e21z229
 - name: Freie Universität Berlin, Germany
   index: 2
   ror: 046ak2485 
date: 13 July 2026
bibliography: paper.bib
---

# Summary

The (Hamiltonian) adaptive resolution simulation ((H-)AdResS) scheme concurrently
couples regions of different resolutions by spatially interpolating or switching between different 
interparticle (potentials/) forces, e.g. atomistic Lennard-Jones and ideal gas [@praprotnik_adaptive_2005; @praprotnik_adaptive_2007; @potestio_hamiltonian_2013]. The method has 
undergone substantial development over the past decades and is now commonly used to simulate open 
atomistic systems exchanging particles and energy with a reservoir, where the microscopic and thermodynamic
states of the reservoir only provide physically consistent boundary conditions for the atomistic
subsystem [@wang_grand-canonical-like_2013; @delle_site_molecular_2019; @cortes-huerto_adaptive_2021]. The framing of the method in terms of the open atomistic system has paved the way towards 
non-equilibrium simulations with distinct thermodynamic reservoir states at opposing boundaries of 
the atomistic region that could even fluctuate in accord with a fluid dynamical simulation on a 
larger scale [@heidari_open-boundary_2020; @klein_nonequilibrium_2021; @gholami_simulation_2022]. However, the method requires very specific algorithms that are not immediately available, 
difficult to maintain and challenging to further develop within the standard simulation packages of 
molecular dynamics. 

# Statement of need

`MRMD` is a stand-alone C++ software package providing the algorithms necessary to set up and run AdResS
simulations. The software provides all algorithms as building blocks that can be combined via a C++ interface. 
It is thus particularly suitable for testing and further 
development of the AdResS method itself. The software comes with basic tools for pre- and postprocessing of 
the simulations but can also parse input and generate output in formats such as GRO and H5MD [@DEBUYL20141546] and thus interfaces 
to standard packages in molecular simulation such as `Gromacs` and `MDAnalysis`.

`MRMD` was designed to be used by researchers in the field of open molecular systems and developers of 
the AdResS method. It has been used in a scientific publication concerned with linking the simulation method 
with the theoretical model of the Liouville-type hierarchy [@hille_reframing_2026] and, in turn, motivating 
algorithmic improvements. The parallel and modular design built around AdResS will facilitate straight-
forward development of the method and help in establishing AdResS as a standard tool of molecular simulation.    

# State of the field                                                                                                                  

Since the establishment of AdResS in the mid-to-late 2000s, the method was implemented 
several times into standard packages of molecular dynamics, e.g. `Espresso++`, `Gromacs`, `Lammps` [@junghans_reference_2010; @fritsch_structure_2012; @heidari_accurate_2016; @Guzman2019ESPResSoPP; @xu2026espressopp]. 
Despite the undeniable research impact of the method, the high cost of maintenance for core functionality exclusive to AdResS applications and the difficulties to separate them from low-level kernels led to instances of discontinued official support, e.g. in the case of `Gromacs`. Furthermore the added complexity and non optimal scaling often results in sub-par performance compared to fully atomistic simulations [@Junghans2017].
This leads to several versions of AdResS being maintained as in-house and closed-source projects.  
This introduced entry barriers for interested outsiders and cumbered further development of the method. 

In light of the growing field of non-equilibrium molecular simulation and the accessibility of software design 
patterns for GPU and multi-node parallelization, AdResS is experiencing an increased interest again. This has 
inspired implementations into packages of molecular dynamics, e.g. `ls1-mardyn` and `Lammps`, that reflect 
the current state of the method [@pinzon_escobar_node-level_2025; @sudhakar_extending_2026]. 

`MRMD`, in contrast, is a stand-alone and open-source software package 
implementing exclusively the AdResS method. Core functionalities specific to AdResS such as the change of molecular 
resolution and the compensation of the associated numerical artifacts are therefore built into its very structure 
and well-covered by unit and integration tests.

# Software design

The design of `MRMD` is heavily influenced by the following three principles: 
1) portability, 2) data-oriented programming, and 3) self-documenting code.
To facilitate widespread use with minimal obstacles `MRMD` uses the Kokkos Performance Portability Library. 
This enables `MRMD` to run on various current and future hardware architectures, including modern state-of-the-art GPUs from Nvidia and AMD.
At the same time, only a single code base needs to be maintained. 
We use CI jobs to continuously monitor the building and execution process for different compilers and architectures.

The software design itself focuses on separating code and data. 
A few data-only containers are used to hold the simulation state: For example, `data::Atoms` holds the positions, velocities, and other information of all atoms within the simulation. 
A separate set of functions takes these data containers as inputs and modifies them in place, e.g. `action::VelocityVerlet`. 
For performance reasons, the data containers are not treated as immutable objects but are modified by the algorithms directly.
All pieces are exposed to the user and can be combined freely to develop new simulation methods.

A precise naming scheme with a clear folder structure, and the avoidance of abbreviations together with literature references within the source code serve as the main point of documentation.
Static code analysis using `clang-tidy` and automatic source code formatting using `clang-format` ensure consistency over the whole code base.
Detailed examples serve as the main entry point for new users and are tested in the CI at every commit to ensure that everything is working correctly and is up to date.

Other third-party libraries are used to extend features or improve usability, namely: `CLI11` for command-line interaction, `YAML-CPP` for config file parsing, `Cabana` [@Slattery2022] for neighbour lists, `HDF5` for IO, and `Googletest` for testing.

# Research impact statement

`MRMD` has 
already been applied as the primary numerical tool of investigation in a study concerned 
with improving the iterative procedure employed in the setup stage of any AdResS simulation
[@hille_reframing_2026]. With regards to the FAIR principles of scientific data management,
the simulations have been integrated into the release version
of `MRMD` as test-covered tutorial scripts and can be reproduced with minimal effort. The 
resulting data has been published to Zenodo[@hille_numerical_2026].

The release version of `MRMD` ships with algorithms for setting up and running AdResS 
with smooth and abrupt changes of resolution for single- and multi-species systems of atomistic 
or molecular composition. The tutorials lead the user towards an AdResS simulation of 
a single-component Lennard-Jones fluid coupled to a reservoir of non-interacting tracer 
particles through abrupt interfaces as it was employed in the aforementioned study.

# Mathematics

We describe here the theory for the abrupt interface AdResS version for which examples are available in the release version of `MRMD`. The smooth interpolation common to other (H-)AdResS flavors is also already implemented in `MRMD`. 

At the heart of AdResS being applied to simulate an open atomistically resolved system in 
exchange with a reservoir through its boundary is the thermodynamic consistency of said reservoir. 

Such a reservoir can be realized in a rectangular simulation domain by the atomistically resolved (AT)
region being encapsulated to the left and right by buffer zones ($\Delta$ regions) within which the 
particles are also interacting atomistically. Beyond the $\Delta$ regions, in the tracer (TR) region, 
the interactions are then abruptly switched off, so that the interaction potential can be written as 

\begin{align}
    U(r, x_i, x_j) = \begin{cases}
            V(r) &\textrm{, for } x_i, x_j \in [-x_{\Delta\textrm{/TR}}, x_{\Delta\textrm{/TR}}] \\
            0 &\textrm{, else } \\
            \end{cases}\textrm{,}
\end{align}

where $x_i$ and $x_j$ are the positions of two particles $i$ and $j$ in $x$-direction, $r$ is the absolute 
distance in between them and

\begin{align}
    V(r) = \begin{cases}
            v(r_{\textrm{cap}}) - \frac{\partial v}{\partial r} \big|_{r_\textrm{cap}} r_{\textrm{cap}} + \frac{\partial v}{\partial r}\big|_{r_\textrm{cap}} r &\textrm{, for } r_{\textrm{cap}} \geq r \\
            v(r) &\textrm{, for } r_{\textrm{cap}} < r \leq r_{\textrm{cut}} \\
            0 &\textrm{, for } r_{\textrm{cut}} < r \\
            \end{cases}\textrm{,}
\end{align}

is a radial potential force-capped beneath a radius $r_{\textrm{cap}}$ with $v(r)$ being a suitable pair-wise potential 
such as the truncated and shifted Lennard-Jones potential. 

It is emphasized that this does not necessarily comprise a Hamiltonian AdResS scheme in the classical 
sense due to the discontinuities of such a potential at $\pm x_{\Delta\textrm{/TR}}$, but that the 
TR region anyways represents merely an algorithm to provide the $\Delta$ regions with the necessary 
number of particles and fluctuations thereof such that they, in turn, can provide the AT region with 
thermodynamically consistent boundary conditions. 

The change of resolution, be it abrupt or smooth, introduces numerical artifacts that can be compensated 
by a one-body thermodynamic force $F_{\textrm{th}}$. This force is calculated during the setup stage of 
AdResS simulations in an iterative procedure 

\begin{align}
    F_{\textrm{th}}^{k + 1}(x) &= F_{\textrm{th}}^{k}(x) - c \nabla \rho^{k}(x) \textrm{,}
\end{align}

where each iteration $k$ comprises a short AdResS simulation with applied thermodynamic force $F_{\textrm{th}}^{k}(x)$, 
which is incremented by the gradient of the density profile $\nabla \rho^{k}(x)$ averaged over this simulation weighted by 
a user-defined convergence prefactor $c$. The procedure is considered converged when the density profile is flat to within 
a desired tolerance. With the converged thermodynamic force, the AdResS production run can be started. 

# AI usage disclosure

Generative AI tools were used in the development of this software, but not in the writing
of this manuscript, or the preparation of supporting materials.

# Acknowledgements

We thank Christoph Junghans and Rupert Klein for insightful discussions, their valuable input, and helpful support. 

We are also grateful to Torsten Stuehn and the Max Planck Institute for Polymer Research for their support and valuable contributions.

Julian F. Hille's contributions to this software have been funded by Deutsche Forschungsgemeinschaft (DFG) through grant CRC 1114 Scaling Cascades in Complex Systems, Project Number 235221301, Project C01 Adaptive coupling of scales in molecular dynamics
and beyond to fluid dynamics.

# References
