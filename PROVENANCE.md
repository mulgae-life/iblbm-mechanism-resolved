# Provenance Notes

## What this repository reproduces

The analysis scripts under `scripts/analysis/` regenerate the locked result tables
(`data/locked_tables/`) byte-identically from the processed data bundled under `data/`,
as stated in the paper's Data Availability Statement. This is the reproducibility
contract of the repository: processed data → tables and figures.

## Simulation code lineage and `config_signature`

The bundled solver (`iblbm/`) and experiment runners (`scripts/experiments/`) are the
maintained versions consistent with the analysis pipeline. The archived processed
simulation outputs bundled under `data/` were generated over the course of the study,
in part with earlier development versions of the same code.

Each run's `status.json` records a `config_signature`: a 16-hex-character SHA-256
prefix over the runner's case-configuration payload, computed by the code version that
generated the run. It is a **legacy configuration fingerprint** — it identifies the
configuration payload as encoded at generation time, and binds neither the output data
bytes nor a source revision.

Because the configuration schema evolved during development, recomputing the signature
with the bundled runner does not reproduce the stored value for 19 of the 63 bundled
sedimentation runs covered by the runner case lists (7 single-particle,
12 two-particle). The difference has been verified field by field for every affected
run (see the machine-readable manifest below):

- **No field present in both payloads changed value.** All differences are schema
  evolution: the legacy field pair (`use_added_mass`, `imc_method`) was unified into
  the current `settling_inertia_model` (`use_added_mass=true` → `explicit_history`;
  otherwise `imc_method` carries the model name), and the fields `xmax` and
  `incompressible_lbgk` were added later with their documented values.
- Under this mapping the internal-mass model, grid, density ratios,
  boundary-enforcement scheme, collision operator, kernel, integrator, and stopping
  rules of every affected run agree with the bundled runner's case definition.

### Machine-readable manifest and verification

- `data/provenance/config_signature_lineage.json` — for each affected run: the path of
  its bundled `status.json`, the stored signature, the legacy configuration payload
  reconstructed from the authors' private development history (its SHA-256 prefix
  reproduces the stored signature), the bundled runner's payload, the field-level
  difference, and `signature_reproducing_revisions` — every tested private-history
  revision whose runner reproduces the stored signature. Where more than one revision
  is listed, those revisions shared the payload schema and the manifest does not
  identify which one executed the run. These identifiers index a private history, are
  not resolvable from this repository, and are recorded for provenance completeness.
- `scripts/analysis/verify_config_signatures.py` — verifies the manifest against
  itself and against the actual bundled `status.json` files, and with
  `--with-runners` additionally cross-checks the bundled runner's payloads and the
  coverage accounting (63 covered runs, mismatch set identical to the manifest). The
  `--with-runners` path imports the solver, so it needs CuPy or the CPU fallback:

```
IBLBM_GPU=0 python3 scripts/analysis/verify_config_signatures.py --with-runners
```

| case_id | stored signature | recomputed by bundled runner |
|---|---|---|
| `SINGLE_RHO150_DF_BGK_VERLET_NONE` | `bbddba13ebbfbb7d` | `4254419b3b170f89` |
| `SINGLE_RHO150_DF_BGK_VERLET_EXPLICIT_HISTORY` | `9e9488f695184bbc` | `ccce20a2352aef95` |
| `SINGLE_RHO150_DF_BGK_VERLET_FENG_B2` | `ddfc07a9abc1535e` | `8e4c80cd8cfc15c5` |
| `SINGLE_RHO150_DF_TRT_VERLET_EXPLICIT_HISTORY` | `9f8405e0159d271c` | `c4b8e134ec907375` |
| `SINGLE_RHO150_DF_CM_MRT_VERLET_EXPLICIT_HISTORY` | `a59269df9cb02ef8` | `0931213b1ce0be8c` |
| `SINGLE_RHO150_MDF_TRT_VERLET_EXPLICIT_HISTORY` | `e7c67b9f7996752e` | `421c63b557d1adda` |
| `SINGLE_RHO150_DFC_TRT_VERLET_EXPLICIT_HISTORY` | `c7bdf6baa1d589cf` | `ac918bd8c6b5294b` |
| `TWO_PARTICLE_DF_BGK_VERLET_NONE` | `a28a4e87186edbb4` | `433872ddf4630d17` |
| `TWO_PARTICLE_DF_BGK_VERLET_EXPLICIT_HISTORY` | `07451e20667ec917` | `31fa3b9cdbdee792` |
| `TWO_PARTICLE_DF_BGK_VERLET_FENG_B2` | `871fa903d10eedb3` | `b859e66dd3ba84db` |
| `TWO_PARTICLE_DF_TRT_VERLET_EXPLICIT_HISTORY` | `1a7613ea777af1de` | `006cbec31344519a` |
| `TWO_PARTICLE_DF_CM_MRT_VERLET_EXPLICIT_HISTORY` | `9f1870662e4b5057` | `9ef214181dc8e94d` |
| `TWO_PARTICLE_MDF_BGK_VERLET_NONE` | `23dfd2aa45967455` | `18e0d06d5d5148d2` |
| `TWO_PARTICLE_MDF_BGK_VERLET_EXPLICIT_HISTORY` | `be2b4d2e3685f090` | `3d252604334d3c76` |
| `TWO_PARTICLE_MDF_TRT_VERLET_EXPLICIT_HISTORY` | `3ef2c98770b10f1d` | `4483d5ddb8036849` |
| `TWO_PARTICLE_DFC_BGK_VERLET_NONE` | `189246f40af04c2e` | `7104ef762dbc36a1` |
| `TWO_PARTICLE_DFC_BGK_VERLET_EXPLICIT_HISTORY` | `31db8d1dc8ed5971` | `1f4ac4ac468451bd` |
| `TWO_PARTICLE_DFC_TRT_VERLET_EXPLICIT_HISTORY` | `de08d8658974552d` | `e70ab9af38490253` |
| `TWO_PARTICLE_REFERENCE_NONE` | `47dcf7c9ed5547f8` | `e7012320dd314864` |

The remaining 44 covered runs recompute to their stored signature with the bundled
runner.

### Warning: re-running the experiment runners over bundled data

The runners' `--skip-completed` logic treats a run as complete only when the stored
signature matches the current one. For the 21 runs above it therefore reports the
bundled outputs as incomplete and, if launched, **re-runs the cases and overwrites the
archived processed outputs** under `data/`. Run the experiment runners only in a
disposable clone of this repository, never in a working copy whose `data/` you intend
to keep. If an overwrite happens unintentionally, review `git status` and restore the
affected run directories from git individually.

## Figure package and the commit cited in the article

The PNGs under `figures/` are byte-identical to the figures of the accepted article,
rendered at 600 dpi (AIP Publishing combination-art resolution). Of the eleven figure
scripts under `scripts/`, three regenerate their panels from the bundled processed
data, five plot the reported table values carried in the script itself, and three
need the optional raw-field package (see the README for the split).

The article's Data Availability Statement cites commit
`a317687d0b33606f4e5aea7dcb756c4b9b761c27`. Git commits are immutable, so that
citation continues to resolve to the repository as it stood at submission. Readers
following that link should be aware that the figure package was corrected afterwards;
the corrections below bring this repository into agreement with the published article
and change no data, no table, and no reported result.

| Item | At the cited commit | Corrected to | Why |
|---|---|---|---|
| `_style.py` output resolution | 300 dpi | 600 dpi | The published figures are 600 dpi; the bundled PNGs were half-resolution copies. |
| Fig. 1 channel-height label | `H` | `L` | The article denotes the channel height `L` (Sec. 2.9.3). |
| Fig. 1 Eulerian grid label | `x_i (Eulerian)` | `x (Eulerian grid)` | The subscript `i` is reserved for the lattice-direction index; the Fig. 1 caption reads "Eulerian grid **x**". |
| Fig. 5 trend overlay | `np.convolve(..., mode="same")` | circular padding, then `mode="valid"` | The azimuth is periodic. Zero-padding outside the array pulled the window mean toward zero and drew a drop at the `theta = 0 / 360` boundary that is not in the data. |
| Fig. 7 annotation placement | Light box aligned above the legend | Light box fixed to the upper-right of the panel | At the cited commit the box overlaid the rising part of the light-particle curve. |
| Fig. 7 abscissa label | `t* = t u_g / D` | `t* = t u_{g,heavy} / D` | The normalisation uses the heavy-particle gravitational velocity. |
| Fig. 8 heavy-particle gap label | `-2.17` | `-2.16` | Table V reports the unrounded recomputation. Recomputing from the two rounded values printed in the figure gives `-2.17`; the script now fixes the table value and says so. |
| Fig. 9 DFC + P4 spread label | `0.22` | `0.21` | Table VI value. |

The reported results are unchanged: no data value, no table entry, and no figure
content differs from the article. Two later changes to the repository are nevertheless
visible in a diff against the cited commit, and are recorded here for completeness.

- **Unused code removed.** Nine solver modules that no case in the article exercises
  (elastic-capsule mechanics, ellipse sedimentation, volume penalization, and the
  Couette, Chen-Wang, and Xia channel boundaries), together with the configuration
  fields and monitor plot branches reachable only from them, were removed. The cases
  the article reports -- fixed cylinder, oscillating cylinder, single- and two-particle
  sedimentation, and Taylor-Green -- build and run unchanged, and the locked tables and
  figures regenerate as described above.
- **Locked-table hash fields updated.** `immutable_table_n_series.json` and
  `n_singles_result.json` record `table_builder_sha256` and `classifier_sha256`, the
  SHA-256 digests of their own generator sources. Editing comments in those generators
  changes the digests, so the two tables were regenerated. The diff against the cited
  commit is confined to those hash fields; every physical value, threshold, window
  definition, and judgment is identical.

## Run-family labels

Labels of the form `N-1`…`N-8`, `G-1`, `G-2` in scripts and in `data/locked_tables/`
are campaign identifiers for run families added at successive stages of the study;
they group related runs and carry no physical meaning. The same applies to the
`a1` and `s44` fragments in bundled filenames such as
`data/two_particle_sedimentation/acceleration_diagnostic/a1_s44_astar_series.json`.
