# IB-LBM Method Reference Map

This table maps each method implemented in `iblbm` to the literature it comes from, and
states how close the implementation is to that source. It is a code-to-reference
correspondence table, not a summary of the cited papers: each row says what the code can
legitimately claim to implement.

## Relationship labels

| Label | Meaning |
|---|---|
| `direct implementation` | The code follows the core algorithm of the cited work. |
| `close lineage` | The idea comes from the cited work, but the time integration, grid, or coupling differs. |
| `benchmark reference` | Cited for the scenario or the comparison values rather than for the algorithm. |
| `project extension` | An option in this code that is not a faithful implementation of any single source. |

## 1. Core LBM, boundary conditions, and IBM

| Item | Code | Author / title | Relationship | Notes |
|---|---|---|---|---|
| LBM forcing | `iblbm/lbm/collision/guo_forcing.py` | Guo, Zheng, Shi, *Discrete lattice effects on the forcing term in the lattice Boltzmann method* (2002) | direct implementation | Guo forcing family. |
| Velocity / pressure BC | `iblbm/boundary/` | Zou, He, *On pressure and velocity flow boundary conditions and bounceback for the lattice Boltzmann BGK model* (1997) | direct implementation | The face relations follow the source; the corner closure is specific to this code. |
| Regularised delta | `iblbm/ibm/common.py` | Peskin, *The immersed boundary method* (2002) | close lineage | Origin of the smoothed delta function. |
| DF | `iblbm/ibm/df.py` | Uhlmann, *An immersed boundary method with direct forcing for the simulation of particulate flows* (2005) | direct implementation | The direct-forcing idea is taken directly; the Navier-Stokes / finite-difference original is transplanted to IB-LBM here. |
| MDF | `iblbm/ibm/mdf.py` | Wang, Fan, Luo, *Combined multi-direct forcing and immersed boundary method for simulating flows with moving particles* (2008) | direct implementation | Core lineage of the iterated forcing. |
| MDF relaxation, A-norm omega | `iblbm/ibm/a_norm.py`, `iblbm/ibm/mdf.py` | Zhang et al., *A relaxed multi-direct-forcing immersed boundary-cascaded lattice Boltzmann method accelerated on GPU* (2020) | direct implementation | The MDF here is Wang (2008) plus the `omega ~ ||A||_inf^-1` estimate of Zhang (2020), Eqs. 39 and 46, and residual-based early termination. |
| DFC | `iblbm/ibm/dfc.py` | Tao et al., *A non-iterative immersed boundary-lattice Boltzmann method with boundary condition enforced for fluid-solid flows* (2019) | direct implementation | Principal reference for the active DFC implementation: the non-iterative distribution-function correction of Eqs. 15-23 and 25. |
| DFC, second-order variant | background only | Tao et al., *A distribution function correction-based immersed boundary-lattice Boltzmann method with truly second-order accuracy for fluid-solid flows* (2022) | close lineage | This code does **not** implement the iterated non-equilibrium / equilibrium DFC of Tao (2022). The article cites the 2022 work only as later literature in the DFC family. |
| BGK | `collision_model="BGK"` | Standard D2Q9 BGK | direct implementation | The core algebra is settled. |
| TRT | `collision_model="TRT"` | Ginzburg, Verhaeghe, d'Humieres, *Two-relaxation-time lattice Boltzmann scheme: about parametrization, velocity, pressure and mixed boundary conditions* (2008) | close lineage | Implemented, but the article does not use it as a benchmark-faithful TRT result. |
| CM-MRT | `collision_model="CM_MRT"` | Geier, Greiner, Korvink, *Cascaded digital lattice Boltzmann automata for high Reynolds number flow* (2006); De Rosis, *Alternative formulation to incorporate forcing terms in a lattice Boltzmann scheme with central moments* (2017) | close lineage | The core relations are implemented; moving-boundary validation is limited. |

## 2. Internal mass, fluid inertia, and correction models

| Item | Code | Author / title | Relationship | Notes |
|---|---|---|---|---|
| Feng internal-mass correction B-2 | `iblbm/physics/inertia/feng_b2.py::apply_imc_correction` | Feng, Michaelides, *Robust treatment of no-slip boundary condition and velocity updating for the lattice Boltzmann simulation of particulate flows* (2009) | direct implementation | Explicit B-2 correction. |
| Internal mass effect | conceptual basis | Suzuki, Inamuro, *Effect of internal mass in the simulation of a moving body by the immersed boundary method* (2011) | close lineage | Frames the internal-mass problem. |
| Improved IBM with internal mass | conceptual basis | Kempe, Frohlich, *An improved immersed boundary method with direct forcing for the simulation of particle laden flows* (2012) | close lineage | MDF plus internal-mass lineage. |
| Second-order IBM, retraction, fluid inertia | `retraction_dx`; basis for `full_volume` | Breugem, *A second-order accurate immersed boundary method for fully resolved simulations of particle-laden flows* (2012) | close lineage | Retraction follows the source directly; the full-volume time discretisation does not. |
| Full-volume / preliminary coupling | `iblbm/physics/inertia/full_volume.py::FullVolumeInertia` | Garcia-Villalba, Fuentes, Dusek, Moriche, Uhlmann, *An efficient method for particle-resolved simulations of neutrally buoyant spheres* (2023) | project extension | This implementation is a surface-IBM plus Eulerian binary-mask hybrid, coupled to `DF` / `MDF` / `DFC` through the IBM strategy. It carries an empirical force scale (`0.58`), so it is not faithful to the source, and multi-particle `DFC` is supported only for particle sets that share `Larea`. Excluded from the evidence base of the article. |
| Added mass, explicit history | `settling_inertia_model="explicit_history"` | Majumder et al., *Computational assessment of immersed boundary-lattice Boltzmann method for complex moving boundary problems* (2023); Feng, Michaelides (2009) | project baseline and extension | The explicit-history path of the main solver is an added-mass treatment with the history term separated out. The added-mass benchmark reading of Majumder and the internal-mass context of Feng (2009) are kept distinct. |
| Rotational degree of freedom, torque | `rotation_half_step()`, `rotation_full_step()`, `extract_torque()` | Uhlmann (2005); Breugem (2012) | close lineage | General lineage for solving particle angular velocity and torque. |
| Rotation benchmark axis | `enable_rotation=True` | Majumder et al. (2023) | benchmark reference | Used for comparing translational and angular velocities; it is not the source of the rotation coupling implemented here. |
| Rotation coupling: indirect | `rotation_coupling="indirect"` | no single source | project extension | Rotates the marker coordinates only. |
| Rotation coupling: semi-implicit | `rotation_coupling="semi_implicit"` | Uhlmann (2005); Breugem (2012); Majumder et al. (2023) | project extension | Applying `Omega x r` from `omega(n)` is useful for benchmark comparison but is not faithful to a single source. |
| Rotation coupling: iterative | `rotation_coupling="iterative"` | Wang, Fan, Luo (2008); Majumder et al. (2023) | project extension | Folding the torque response into the MDF iteration is specific to this code. |

## 3. Scenario and benchmark references

| Scenario | Code | Author / title | Relationship | Notes |
|---|---|---|---|---|
| Fixed cylinder | `scenarios/steady.py` | Kang, Hassan, *A comparative study of direct-forcing immersed boundary-lattice Boltzmann methods for stationary complex boundaries* (2011) | benchmark reference | Comparison axis for stationary boundaries. |
| Fixed-cylinder Dirichlet-Neumann BC | `iblbm/boundary/dirichlet_neumann.py::apply_bc_dirichlet_neumann` | Kang, Hassan (2011) | benchmark reference | Dedicated closure for Dirichlet on left/top/bottom and homogeneous Neumann on the right. |
| Oscillating cylinder | `scenarios/oscillating.py` | Dutsch et al., *Low-Reynolds-number flow around an oscillating circular cylinder at low Keulegan-Carpenter numbers* (1998) | benchmark reference | Experimental anchor. |
| Oscillating / moving-boundary IB-LBM | comparison axis of `scenarios/oscillating.py` | Uhlmann (2005); Tao et al. (2019); Majumder et al. (2023) | benchmark reference | The preset carries no benchmark-faithful assertion. |
| Canonical single-particle sedimentation | `scenarios/sedimentation.py::make_sedimentation_config` | Glowinski et al., *A fictitious domain approach ... application to particulate flow* (2001) | benchmark reference | Named for the Glowinski anchor; the implementation is not the FEM fictitious-domain method. |
| Single-particle MDF benchmark | comparison axis of the same generator | Wang, Fan, Luo (2008) | benchmark reference | Geometry and physics anchor. |
| Single-particle reference benchmark | `scenarios/sedimentation.py::make_single_particle_config_reference_benchmark` | Wang, Fan, Luo (2008) | benchmark reference | Generator that pins `h`, `dt`, and `D/h` to the published values. |
| Single-particle IB-LBM improvement axis | comparison axis of the same generator | Feng, Michaelides (2009); Tao et al. (2019) | benchmark reference | This code is a project-specific hybrid. |
| Two-particle pure wake interaction | `scenarios/sedimentation.py::make_two_particle_config` | Uhlmann (2005) | benchmark reference | Only the coordinates are taken from the source; the numerical setup differs substantially. |
| Two-particle reference benchmark | `scenarios/sedimentation.py::make_two_particle_config_reference_benchmark` | Majumder et al. (2023) | benchmark reference | Benchmark generator for the main solver. Defaults: `BGK + DF + euler_explicit + explicit_history + incompressible_lbgk=True`. |

## 4. Options that need a careful reading

| Option | Status |
|---|---|
| `settling_inertia_model="full_volume"` | One candidate treatment of added and internal mass, but not faithful to any single source. It couples surface-IBM force extraction (`DF` / `MDF` / `DFC`) with an Eulerian full-volume internal-inertia integral, and carries an empirical force scale (`0.58`), so it is an exploratory path. Multi-particle `DFC` is supported only for particle sets that share `Larea`. Excluded from the evidence base of the article and retained for traceability. |
| `settling_inertia_model="explicit_history"` | The explicit-history added-mass path of the main solver. |
| `mdf_iterations`, `mdf_min_iterations`, `mdf_tolerance` | MDF here is not the fixed-iteration scheme of Wang (2008) alone: it adds the `omega` estimate of Zhang (2020) and residual-based early termination. Setting `min = max` in a benchmark runner turns part of the adaptive behaviour off. |
| `rotation_coupling in {"indirect", "semi_implicit", "iterative"}` | Used for benchmark comparison; none is a faithful implementation of a fixed source. |
| `make_sedimentation_config()` | Generator inspired by the Glowinski anchor. |
| `make_two_particle_config()` | Generator inspired by Uhlmann. |
