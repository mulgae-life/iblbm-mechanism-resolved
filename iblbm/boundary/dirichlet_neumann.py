"""Kang-Hassan 고정 실린더 benchmark용 Dirichlet + Neumann 혼합 경계.

Face 구성
  ┌──────────── top = Dirichlet (u_in, 0)  ─────────────┐
  │                                                       │
  left                                                    right
  Dirichlet (u_in, 0)               homogeneous Neumann (∂/∂n = 0)
  │                                                       │
  └────────── bottom = Dirichlet (u_in, 0) ───────────────┘

알고리즘
  - Dirichlet faces (left/top/bottom): prescribed (u_in, 0) + ρ = ρ_ref
      → `f = f^eq(ρ, u)`로 equilibrium 초기화 (Dirichlet 강제)
  - Neumann face (right): `right_inner = right − 1`의 post-streaming 분포 재사용
      · ρ, u는 right_inner에서 macroscopic 복원 (∂ρ/∂x = ∂u/∂x = 0 근사)
      · f_right = f^eq(ρ, u) + (f_{right_inner} − f^eq_{right_inner})
        → non-equilibrium 성분을 내부 노드에서 복사하는 외삽형

계보
  - face 역할 분배: Kang & Hassan (2011) §3.2.1
        "For inlet and far-field boundaries, the Dirichlet boundary condition
         is used and for the outlet boundary, the homogeneous Neumann boundary
         condition is used."
  - 단일 논문을 그대로 따른 closure는 아님 (equilibrium Dirichlet + non-equilibrium 외삽 조합은 프로젝트 구현)
  - 참고: 일반 face closure는 Zou & He (1997) §3 계보

Corner
  - left corners (top-left, bottom-left): Dirichlet (u_in, 0) + equilibrium refill
  - right corners (top-right, bottom-right): 대각 내부 노드(`right_diag`)에서 Neumann 복사
"""

from __future__ import annotations

from ..backend import xp as np
from ..lbm import compute_feq
from .base import BoundaryIndices, register_boundary_as


def apply_bc_dirichlet_neumann(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    inflow_u: float, dens: float, idx: BoundaryIndices,
    lattice=None,
) -> None:
    """Kang-Hassan benchmark BC 적용.

    단계
      1. left/top/bottom: Dirichlet (u_in, 0) + ρ = dens + f ← f^eq
      2. right (Neumann): right_inner에서 (ρ, u) 복원 → f_right = f^eq + (f_inner − f_inner^eq)
      3. left corners: Dirichlet + equilibrium refill
      4. right corners: right_diag에서 Neumann 복사

    참조
      - Kang & Hassan (2011) §3.2.1 — Dirichlet inflow/far-field + homogeneous Neumann outlet 역할 분배
      - 단일 논문을 그대로 따른 closure는 아님. equilibrium Dirichlet + non-equilibrium 외삽 조합은 프로젝트 구현
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx = idx.nx
    ny = idx.ny
    ex = np.asarray(lattice.e[:, 0])
    ey = np.asarray(lattice.e[:, 1])

    def _macro_from_streamed(nodes):
        f_loc = fstar[nodes]
        ro_loc = np.sum(f_loc, axis=1)
        U_loc = np.zeros((len(nodes), 2))
        U_loc[:, 0] = np.sum(f_loc * ex[None, :], axis=1) / ro_loc
        U_loc[:, 1] = np.sum(f_loc * ey[None, :], axis=1) / ro_loc
        return ro_loc, U_loc

    right_inner = right - 1

    ro[left] = dens
    U[left, 0] = inflow_u
    U[left, 1] = 0.0
    fstar[left] = compute_feq(ro[left], U[left], lattice)

    ro[top] = dens
    U[top, 0] = inflow_u
    U[top, 1] = 0.0
    fstar[top] = compute_feq(ro[top], U[top], lattice)

    ro[bottom] = dens
    U[bottom, 0] = inflow_u
    U[bottom, 1] = 0.0
    fstar[bottom] = compute_feq(ro[bottom], U[bottom], lattice)

    ro_right, U_right_inner = _macro_from_streamed(right_inner)
    ro[right] = ro_right
    U[right] = U_right_inner
    fstar[right] = compute_feq(ro[right], U[right], lattice) + (
        fstar[right_inner] - compute_feq(ro_right, U_right_inner, lattice)
    )

    left_corners = np.array([
        0,
        nx * (ny - 1),
    ])
    right_corners = np.array([
        nx - 1,
        nx * ny - 1,
    ])
    right_diag = np.array([
        2 * nx - 2,
        nx * (ny - 1) - 2,
    ])

    ro[left_corners] = dens
    U[left_corners, 0] = inflow_u
    U[left_corners, 1] = 0.0
    fstar[left_corners] = compute_feq(ro[left_corners], U[left_corners], lattice)

    ro_right_corner, U_right_corner_inner = _macro_from_streamed(right_diag)
    ro[right_corners] = ro_right_corner
    U[right_corners] = U_right_corner_inner
    fstar[right_corners] = compute_feq(ro[right_corners], U[right_corners], lattice) + (
        fstar[right_diag] - compute_feq(ro_right_corner, U_right_corner_inner, lattice)
    )


class DirichletNeumannBC:
    name = "dirichlet_neumann"

    def apply(self, fstar, ro, U, state, cfg, ttt: int) -> None:
        apply_bc_dirichlet_neumann(fstar, ro, U, state.in_u, state.dens, state.idx, state.lattice)

_inst = DirichletNeumannBC()
register_boundary_as(_inst, "dirichlet_neumann")
