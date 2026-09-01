"""진동 실린더 benchmark 시나리오 builder.

Dütsch et al. (1998) Re=100, KC=5 비교축을 기준으로 구성한다.
"""

from __future__ import annotations

from iblbm.config import SimConfig
from iblbm.solver import run


def make_oscillating_cylinder_config(
    *,
    reynolds: float = 100.0,
    kc: float = 5.0,
    ibm_method: str = "DF",
    delta_type: str = "peskin4pt",
    collision_model: str = "BGK",
    nn: int = 1601,
    max_steps: int = 40_000,
) -> SimConfig:
    return SimConfig(
        Re=float(reynolds),
        NN=int(nn),
        xmax=1.5,
        ymax=1.0,
        cylinder_center=(0.75, 0.5),
        cylinder_D_ratio=1.0 / 20.0,
        lattice_u=0.1,
        inflow_u=0.0,
        bc_type="open_boundary",
        delta_type=delta_type,
        ibm_method=ibm_method,
        collision_model=collision_model,
        mdf_iterations=20 if ibm_method == "MDF" else 1,
        mdf_min_iterations=5 if ibm_method == "MDF" else 1,
        max_steps=int(max_steps),
        check_interval=50,
        use_convergence=False,
        motion_type="oscillating",
        KC=float(kc),
        marker_spacing_factor=2.0 / 3.0,
        nx_formula="simple",
    )


cfg = make_oscillating_cylinder_config()


if __name__ == "__main__":
    run(cfg)
