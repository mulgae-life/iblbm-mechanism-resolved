"""고정 실린더 benchmark 시나리오 builder.

Dirichlet-Neumann benchmark BC를 기준으로 Re20/40/100/200 고정 실린더
실험 세트를 구성한다. Re40/100은 Kang & Hassan (2011) 비교축이고,
Re20/200은 동일 BC/도메인 family 확장으로 사용한다.
"""

from __future__ import annotations

from iblbm.config import SimConfig
from iblbm.solver import run


def make_dirichlet_neumann_fixed_cylinder_config(
    *,
    reynolds: float = 40.0,
    ibm_method: str = "DF",
    delta_type: str = "hat",
    collision_model: str = "BGK",
    max_steps: int | None = None,
    mdf_iterations: int | None = None,
    mdf_min_iterations: int | None = None,
    nn: int | None = None,
) -> SimConfig:
    re = float(reynolds)
    if re in {20.0, 40.0}:
        xmax = 1.0
        default_nn = 1601
        default_steps = 100_000
    elif re == 100.0:
        xmax = 1.25
        default_nn = 1601
        default_steps = 100_000
    elif re == 200.0:
        xmax = 1.25
        # NN=2401 → D=60Δx, ν=0.03, τ=0.59 (BGK 안전; 가드 임계 0.55 대비 마진 0.04).
        # NN=1601 (τ=0.56)은 가드 임계 마진 0.01뿐이라 shedding 진동에서 불안정 위험으로 채택 안 함.
        # NN=3201 (τ=0.62) 보수 마진은 계산 비용 (~80h)이 커서 생산 격자는 NN=2401을 사용.
        # max_steps=60000 → 19 vortex shedding cycle (T_shed≈3158 step).
        # 후처리는 50% tail (step 30000~60000, 9.5 cycle) 평균을 사용해 계산 시간을 ~24h 줄인다.
        default_nn = 2401
        default_steps = 60_000
    else:
        raise ValueError(
            f"Unsupported fixed-cylinder Reynolds number: {re}. "
            "Allowed: 20, 40, 100, 200."
        )

    resolved_nn = default_nn if nn is None else int(nn)
    resolved_steps = default_steps if max_steps is None else int(max_steps)
    if ibm_method == "MDF":
        resolved_mdf_iterations = 20 if mdf_iterations is None else int(mdf_iterations)
        resolved_mdf_min_iterations = 5 if mdf_min_iterations is None else int(mdf_min_iterations)
    else:
        resolved_mdf_iterations = 1
        resolved_mdf_min_iterations = 1

    return SimConfig(
        Re=re,
        NN=resolved_nn,
        xmax=xmax,
        ymax=1.0,
        cylinder_center=(0.5, 0.5),
        cylinder_D_ratio=1.0 / 40.0,
        lattice_u=0.1,
        inflow_u=0.1,
        bc_type="dirichlet_neumann",
        delta_type=delta_type,
        ibm_method=ibm_method,
        collision_model=collision_model,
        mdf_iterations=resolved_mdf_iterations,
        mdf_min_iterations=resolved_mdf_min_iterations,
        max_steps=resolved_steps,
        check_interval=200,
        use_convergence=False,
        cd_convergence_tol=1e-4,           # 최근 5개 샘플 범위 < 1e-4 시 조기 종료
        cd_convergence_window=5,
        cd_convergence_start_step=20_000,  # 초기 transient 회피
        marker_spacing_factor=2.0 / 3.0,
        nx_formula="standard",
    )


cfg = make_dirichlet_neumann_fixed_cylinder_config()


if __name__ == "__main__":
    result = run(cfg)
    print(f"\nFinal Cd = {result['Cd_history'][-1]:.6f}")
    print(f"Final Cl = {result['Cl_history'][-1]:.6f}")
