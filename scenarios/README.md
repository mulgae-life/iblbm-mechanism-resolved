# Scenarios

`scenarios/` provides user-facing preset and configuration builders.
The time-loop orchestration lives in `iblbm/runtime/scenarios/`.

| File | Role | Class | Notes |
|---|---|---|---|
| `steady.py` | Fixed-cylinder Dirichlet-Neumann benchmark builder | benchmark | Re20/40/100/200 fixed-cylinder cases |
| `oscillating.py` | Oscillating-cylinder Re100 KC5 builder | benchmark | Oscillating-cylinder case |
| `sedimentation.py` | Single- and two-particle sedimentation config generator | benchmark | Sedimentation cases |
| `taylor_green.py` | Taylor-Green decaying-vortex config generator | verification | Spatial convergence check (Fig. D.1) |

Principles:
- Active benchmark families are fixed cylinder, oscillating cylinder, single-particle sedimentation, and two-particle sedimentation.
- Taylor-Green is a verification case, not a benchmark family: it measures the spatial convergence order reported in Appendix D.
- Add user presets here, but keep runtime dispatch in `iblbm/runtime/scenarios/`.
