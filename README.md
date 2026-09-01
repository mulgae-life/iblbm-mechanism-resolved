# Mechanism-Resolved Interface Momentum Transfer in Immersed-Boundary Lattice Boltzmann Simulations

This repository contains the source code, selected processed benchmark data,
and final figures supporting the paper:

> **H. Jo**, "Mechanism-Resolved Interface Momentum Transfer in Immersed-Boundary Lattice Boltzmann Simulations," *Physics of Fluids* (accepted 2026), https://doi.org/10.1063/5.0336036.
>
> The DOI resolves once the article is published.

The benchmark cases are used as controlled probes of four interface
momentum-transfer mechanisms: prescribed-body local no-slip fidelity,
DFC correction redistribution under kernel choice, configuration-dependent sensitivity to the numerical
internal/fictitious-fluid force-evaluation closure, and collision-model controls
(BGK / TRT / CM-MRT). Three direct-forcing IB-LBM schemes (DF, MDF, DFC)
are implemented within a common solver.

## Repository contents

| Directory | Description |
|-----------|-------------|
| `iblbm/` | Core IB-LBM solver (LBM + IBM modules). `iblbm/METHOD_REFERENCE_MAP.md` maps each implemented method to its reference |
| `scenarios/` | Benchmark scenario definitions (steady, oscillating, sedimentation, Taylor-Green) |
| `scripts/` | Figure scripts, shared style/path helpers, and analysis generators (`scripts/analysis/`) for the locked tables, N-series confirmations, phase decomposition, and single-particle transient metrics |
| `data/` | Selected processed outputs (`status.json`, sedimentation histories, reference CSVs, Taylor-Green convergence arrays) |
| `figures/` | Final rendered paper figures (Fig. 1–9, S1, D1; 11 files), byte-identical to the accepted article at 600 dpi |
| `monitor/` | Run-time callback used by the experiment runners: it writes the sedimentation history and run status (`sedimentation_history.json`, `status.json`) and renders progress plots |

## Three boundary-enforcement schemes

- **DF** (Direct Forcing) — Peskin-type explicit forcing
- **MDF** (Multi-Direct Forcing) — Adaptive iterative correction (N ∈ [5, 20])
- **DFC** (Distribution Function Correction) — Post-collision distribution correction

Each scheme is tested with two delta-function kernels: hat (1D support 2h) and Peskin 4-point (1D support 4h, P4).

## Requirements

```bash
pip install -r requirements.txt
```

The solver runs on the GPU by default and requires CuPy; set `IBLBM_GPU=0` to run it on
the CPU instead. The figure scripts need neither. Three of the analysis commands do:
`build_immutable_table.py`, `n_singles_confirm.py`, and `verify_config_signatures.py
--with-runners` rebuild the runner case configurations and therefore import the solver.
They are shown below with the `IBLBM_GPU=0` prefix; drop it if CuPy is installed.

## Data structure

Every directory below backs a specific table or figure of the article.

```
data/
├── fixed_cylinder/          # Fixed-cylinder matrix: Re = 20/40/100/200 x {DF, MDF, DFC}
│   │                        #   x {hat, P4} x {BGK, TRT, CM-MRT}, one status.json each.
│   │                        #   Tables I, II, VI(a), B.4 and Supplementary S1-S3; Figs. 2, 3, 4, 9(a)
│   └── re200/re200_summary_50tail.json      # 50% tail post-processing for the Re=200 row
├── oscillating_cylinder/    # Re=100, KC=5 matrix in the same layout (Table B.8)
│   ├── grid_sensitivity/    #   DF/BGK/P4 grid series, NN = 801, 1281, 1601 (Appendix C.4, Supplementary S2.3)
│   └── dutsch_reference/    #   Digitised LDA velocity profiles from Dutsch et al. (1998)
├── taylor_green/            # Spatial convergence arrays per scheme (Tables D.1 and D.2, Fig. D.1)
├── grid_sensitivity/        # Fixed-cylinder grid studies: Re=40 (NN = 801, 1601), and DF/P4 sweeps
│                            #   at Re=100 (N_y = 641-1601, with two DFC runs at 1601) and Re=200
│                            #   (N_y = 961-1921). Supplementary Table S5, Appendix C.3
├── single_particle_sedimentation/          # Status files and scalar histories: baseline/, and a
│                                           #   method/kernel/collision matrix per density ratio
│                                           #   (BGK at rho = 1.01, 1.1, 1.25 plus a reduced-viscosity
│                                           #   rho = 1.25 set; BGK/TRT/CM-MRT at rho = 1.5), with
│                                           #   explicit-history / without-correction ablation pairs,
│                                           #   a marker-retraction series, and grid pairs.
│                                           #   Tables IV, B.5, VII, Supplementary S10, Appendix C.5
├── single_particle_sedimentation_tall60D/  # 60D extended channel, rho = 1.5 (Tables C.1, C.2)
├── single_particle_sedimentation_tall80D/  # 80D extended channel, rho = 1.5, DF (Table VII)
├── two_particle_sedimentation/   # Wake-interaction trajectories and Reynolds histories, the coupled
│   │                             #   space-time grid-refinement pairs (N_y = 961, 1281), 60D/80D
│   │                             #   extensions, and isolated-light controls.
│   │                             #   Tables V, VI(b), B.7, C.3; Figs. 7 and 9(b)
│   └── acceleration_diagnostic/  #   Particle-bound acceleration proxy |a*(t)| (Velocity-Verlet pairs)
├── locked_tables/           # Locked analysis tables regenerated by scripts/analysis/
└── provenance/              # config_signature lineage manifest (see PROVENANCE.md)
```

## Reproducing figures

The final PNG figures are included in `figures/`. The eleven scripts fall into
three groups.

**Read the bundled data** — run these directly:

```bash
python scripts/make_fig2_cd_comparison.py     # steady-cylinder drag histories
python scripts/make_fig4_delta_sensitivity.py # kernel sensitivity
python scripts/make_fig7_sed_trajectory.py    # two-particle trajectory and Reynolds history
```

**Plot the values as reported in the article** — Figs. 3, 8, 9, D.1 and the
domain schematic (Fig. 1) carry the table entries as literals in the script and
need no input file. The tables they draw from are themselves reproducible: see
the locked-table section below and `data/taylor_green/` for the Taylor-Green
convergence arrays behind Fig. D.1.

**Need the raw fields** — Figs. 5, 6 and S1 draw velocity fields and
marker-resolved DFC diagnostics. Their scripts keep the full analysis logic but
expect an optional `data/raw_fields/` package that is not bundled here; running
them prints which file is missing. The rendered PNGs in `figures/` are the
published versions.

## Reproducing the locked tables

The locked analysis tables under `data/locked_tables/` regenerate byte-identically
from the bundled processed data. Each generator writes its output next to itself
in `scripts/analysis/` (those paths are git-ignored); compare the result against
the bundled copy under `data/`.

```bash
# Phase-resolved paired-difference decomposition
#   -> scripts/analysis/a1_phase_result.json
#   -> scripts/analysis/a1_s44_astar_series.json
python scripts/analysis/a1_phase_decomposition.py

# Supplementary Table S10 — single-particle transient metrics on the three
# production explicit-history baselines (terminal / peak / Delta% / t*_99)
python scripts/analysis/table_s10_production.py

# N-series immutable table, confirmatory path (both flags are required)
#   -> scripts/analysis/immutable_table_n_series.json
IBLBM_GPU=0 python scripts/analysis/build_immutable_table.py \
    --confirmatory --contact-rule-confirmed CENTER_DISTANCE_LE_D_V1

# N-3 / N-5 / N-8 single-run confirmations, including the N-2 block
#   -> scripts/analysis/n_singles_result.json
IBLBM_GPU=0 python scripts/analysis/n_singles_confirm.py --with-n2
```

Both flags are load-bearing:

- Without `--confirmatory`, `build_immutable_table.py` runs its self-test and
  exits without writing a table. The confirmatory path additionally requires the
  contact-rule identifier to be passed explicitly, so that the rule is fixed
  before any result is produced.
- Without `--with-n2`, `n_singles_confirm.py` omits the `n2` block, and the
  output then does not match the bundled table.

To check byte-identity against the bundled copies:

```bash
cmp scripts/analysis/a1_phase_result.json \
    data/locked_tables/a1_phase_result.json
cmp scripts/analysis/a1_s44_astar_series.json \
    data/two_particle_sedimentation/acceleration_diagnostic/a1_s44_astar_series.json
cmp scripts/analysis/immutable_table_n_series.json \
    data/locked_tables/immutable_table_n_series.json
cmp scripts/analysis/n_singles_result.json \
    data/locked_tables/n_singles_result.json
```

## Provenance and re-running the experiment runners

Code-lineage notes for the bundled runs are in [PROVENANCE.md](PROVENANCE.md).
In particular, 19 of the bundled sedimentation runs carry a `config_signature`
written by an earlier development version of the code; the bundled runners
recompute a different value for them (schema evolution only — no configuration
field changed value; see the manifest under `data/provenance/` and
`scripts/analysis/verify_config_signatures.py`).

**Warning:** because of this, the runners' `--skip-completed` logic treats
those 19 runs as incomplete and, if launched, re-runs them and overwrites the
archived outputs under `data/`. Run the experiment runners only in a disposable
clone, never in a working copy whose `data/` you intend to keep.

## What is excluded

Large raw field outputs (`velocity_field.npz`, DFC diagnostic arrays, and
videos) are excluded due to file size. These are available from the
corresponding author upon reasonable request. The two-particle trajectory and
Reynolds-history scalar time series are bundled under
`data/two_particle_sedimentation/`.

## Author

- **Hongju Jo** — Independent Researcher, Seoul, Republic of Korea
- Correspondence: jhjoo3217@yonsei.ac.kr

## License

MIT License — see [LICENSE](./LICENSE).
