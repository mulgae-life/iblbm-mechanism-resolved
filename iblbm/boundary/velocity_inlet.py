"""Velocity inlet + pressure outlet + no-slip top/bottom 경계.

Face 구성
  - left    prescribed u = (u_in, 0),   ρ는 known population으로 복원
  - right   prescribed ρ = ρ_out,       u_x는 known population으로 복원, u_y는 내부 외삽
  - top     no-slip u = 0,              접선 u_x는 내부 외삽
  - bottom  no-slip u = 0,              접선 u_x는 내부 외삽

D2Q9 방향 + left face 미지 population
        6 ─── 2 ─── 5
         ╲   │   ╱
      3 ─── 0 ─── 1           known   = {f_0, f_2, f_3, f_4, f_6, f_7}
         ╱   │   ╲            unknown = {f_1, f_5, f_8}   (+x 방향, 유체→내부)
        7 ─── 4 ─── 8

Left face 식 (Zou & He 1997 좌측 inlet 회전; 원 논문은 bottom face 기준 Eq. (17-18))
  - 밀도 복원
        ρ = [f_0 + f_2 + f_4 + 2(f_3 + f_6 + f_7)] / (1 − u_x)
  - 미지 population (non-equilibrium bounceback)
        f_1 = f_3 + (2/3) ρ u_x
        f_5 = f_7 − ½ (f_2 − f_4) + (1/6) ρ u_x + ½ ρ u_y
        f_8 = f_6 + ½ (f_2 − f_4) + (1/6) ρ u_x − ½ ρ u_y

Right face (압력 outlet): ρ = ρ_out 고정, u_x는 known population으로 복원
  - u_x = −1 + [f_0 + f_2 + f_4 + 2(f_1 + f_5 + f_8)] / ρ_out
  - 미지 {f_3, f_6, f_7}는 대칭 분해 식 (Eq. (23) 반사형)
  - u_y는 내부 2-점 외삽 (4 u_{j,i−1} − u_{j,i−2}) / 3 (프로젝트 구현)

계보
  - 직선 face Zou-He closure: Zou & He (1997) §3 Eq. (17-18, 22-23)

프로젝트 구현
  - top/bottom 접선 속도 2차 외삽
  - inlet/outlet corner에 대한 ρ/u 재설정 + equilibrium 초기화
"""

from __future__ import annotations

from ..backend import xp as np
from ..lbm import compute_feq
from .base import register_boundary
from .base import BoundaryIndices


def apply_bc_velocity_inlet(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    inflow_u: float, dens: float, idx: BoundaryIndices,
    lattice=None,
) -> None:
    """Velocity inlet + pressure outlet + no-slip top/bottom 경계 적용.

    수식
      - left (Zou-He Eq. (17-18)): ρ = Σ_known/(1−u_x), f_1/f_5/f_8 복원
      - right (Zou-He Eq. (22-23) 반사형): u_x 복원, f_3/f_6/f_7 복원, u_y는 내부 외삽
      - top/bottom: no-slip face (Zou-He 반사형), 접선 u_x는 내부 외삽

    Corner (lattice != None)
      - inlet corners: prescribed (u_in, 0) + ρ = ρ_out + `compute_feq`
      - outlet corners: u_x 2-점 외삽 + u_y = 0 + ρ = ρ_out + `compute_feq`

    참조
      - Zou & He (1997) §3 — face별 미지 population closure
      - Corner refill + 접선 외삽은 프로젝트 구현
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx = idx.nx

    U[left, 0] = inflow_u
    U[left, 1] = 0.0
    ro[left] = (1.0 / (1.0 - U[left, 0])) * (
        fstar[left, 0] + fstar[left, 2] + fstar[left, 4]
        + 2.0 * (fstar[left, 3] + fstar[left, 6] + fstar[left, 7])
    )
    fstar[left, 1] = fstar[left, 3] + (2.0 / 3.0) * ro[left] * U[left, 0]
    fstar[left, 5] = (
        fstar[left, 7]
        - 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        + 0.5 * ro[left] * U[left, 1]
    )
    fstar[left, 8] = (
        fstar[left, 6]
        + 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        - 0.5 * ro[left] * U[left, 1]
    )

    ro[right] = dens
    U[right, 0] = -1.0 + (
        fstar[right, 0] + fstar[right, 2] + fstar[right, 4]
        + 2.0 * (fstar[right, 1] + fstar[right, 5] + fstar[right, 8])
    ) / ro[right]
    U[right, 1] = (4.0 * U[right - 1, 1] - U[right - 2, 1]) / 3.0
    fstar[right, 3] = fstar[right, 1] - (2.0 / 3.0) * ro[right] * U[right, 0]
    fstar[right, 7] = (
        fstar[right, 5]
        + 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        - 0.5 * ro[right] * U[right, 1]
    )
    fstar[right, 6] = (
        fstar[right, 8]
        - 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        + 0.5 * ro[right] * U[right, 1]
    )

    U[top, 0] = (4.0 * U[top - nx, 0] - U[top - 2 * nx, 0]) / 3.0
    U[top, 1] = 0.0
    ro[top] = (1.0 / (1.0 + U[top, 1])) * (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    )
    fstar[top, 4] = fstar[top, 2] - (2.0 / 3.0) * ro[top] * U[top, 1]
    fstar[top, 7] = (
        fstar[top, 5]
        + 0.5 * (fstar[top, 1] - fstar[top, 3])
        - 0.5 * ro[top] * U[top, 0]
    )
    fstar[top, 8] = (
        fstar[top, 6]
        - 0.5 * (fstar[top, 1] - fstar[top, 3])
        + 0.5 * ro[top] * U[top, 0]
    )

    U[bottom, 0] = (4.0 * U[bottom + nx, 0] - U[bottom + 2 * nx, 0]) / 3.0
    U[bottom, 1] = 0.0
    ro[bottom] = (1.0 / (1.0 - U[bottom, 1])) * (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    )
    fstar[bottom, 2] = fstar[bottom, 4] + (2.0 / 3.0) * ro[bottom] * U[bottom, 1]
    fstar[bottom, 5] = (
        fstar[bottom, 7]
        - 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + 0.5 * ro[bottom] * U[bottom, 0]
    )
    fstar[bottom, 6] = (
        fstar[bottom, 8]
        + 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        - 0.5 * ro[bottom] * U[bottom, 0]
    )

    if lattice is not None:
        inlet_corners = np.array([left[0], left[-1]])
        ro[inlet_corners] = dens
        U[inlet_corners, 0] = inflow_u
        U[inlet_corners, 1] = 0.0
        fstar[inlet_corners] = compute_feq(ro[inlet_corners], U[inlet_corners], lattice)

        outlet_corners = np.array([right[0], right[-1]])
        ro[outlet_corners] = dens
        for ci in outlet_corners:
            U[ci, 0] = (4.0 * U[ci - 1, 0] - U[ci - 2, 0]) / 3.0
        U[outlet_corners, 1] = 0.0
        fstar[outlet_corners] = compute_feq(ro[outlet_corners], U[outlet_corners], lattice)


class VelocityInletBC:
    name = "velocity_inlet"

    def apply(self, fstar, ro, U, state, cfg, ttt: int) -> None:
        apply_bc_velocity_inlet(fstar, ro, U, state.in_u, state.dens, state.idx, state.lattice)


register_boundary(VelocityInletBC())
