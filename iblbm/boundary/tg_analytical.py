"""Taylor-Green 해석해 경계 (time-dependent Dirichlet).

Face 구성
  - 4-face 모두 해석해 u(x, y, t) 강제
      · 법선 속도: Zou-He 압력형 closure로 ρ 복원
      · 미지 population: Zou-He Eq. (23) 반사형
  - corner 4개: 해석해 u + ρ = 1 + `compute_feq` 재설정

해석해 (감쇠 Taylor-Green vortex)
  - u_x = −u_0 cos(π x/L) sin(π y/L) exp(−2ν(π/L)² t)
  - u_y =  u_0 sin(π x/L) cos(π y/L) exp(−2ν(π/L)² t)
    · 도메인 좌표 [−L, L]²
    · `tg_analytical_velocity_field(X, Y, t, u0, L, ν)`가 위 식을 반환

계보
  - 해석해 본문: Taylor-Green 표준 감쇠 와류 형태
  - face closure: Zou & He (1997) §3 velocity wall BC 회전형
  - corner reset은 프로젝트 구현
"""

from __future__ import annotations

from ..backend import xp as np
from ..diagnostics import tg_analytical_velocity_field
from ..lbm import compute_feq
from .base import BoundaryIndices, register_boundary


def apply_bc_analytical(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    t_physical: float, cfg, idx: BoundaryIndices, lattice,
) -> None:
    """Taylor-Green time-dependent Dirichlet 경계 적용.

    단계
      1. 4-face 모두에 대해
            · 노드별 도메인 좌표 (x, y) 계산 (중심 원점화)
            · 해석해 `(u_x, u_y) = tg_analytical_velocity_field(x, y, t, u_0, L, ν)`
            · ρ를 Zou-He 압력형 closure로 복원
            · 미지 population을 Zou-He Eq. (23) 반사형으로 복원
      2. corner 4개에 해석해 u + ρ = 1 + `compute_feq` refill

    참조
      - Zou & He (1997) §3 face closure
      - Corner reset은 프로젝트 구현
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx, ny = idx.nx, idx.ny

    L_lat = 0.5 * (cfg.NN - 1)
    lattice_D = cfg.cylinder_D_ratio * (cfg.NN - 1)
    nu = cfg.lattice_u * lattice_D / cfg.Re

    X_lat = np.arange(nx, dtype=float) - (nx - 1) / 2.0
    Y_lat = np.arange(ny, dtype=float) - (ny - 1) / 2.0

    j_left = left // nx
    x_left = np.full_like(j_left, X_lat[0], dtype=float)
    y_left = Y_lat[j_left]
    ux_t, uy_t = tg_analytical_velocity_field(x_left, y_left, t_physical, cfg.tg_u0, L_lat, nu)
    U[left, 0] = ux_t
    U[left, 1] = uy_t
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

    j_right = right // nx
    x_right = np.full_like(j_right, X_lat[nx - 1], dtype=float)
    y_right = Y_lat[j_right]
    ux_t, uy_t = tg_analytical_velocity_field(x_right, y_right, t_physical, cfg.tg_u0, L_lat, nu)
    U[right, 0] = ux_t
    U[right, 1] = uy_t
    ro[right] = (1.0 / (1.0 + U[right, 0])) * (
        fstar[right, 0] + fstar[right, 2] + fstar[right, 4]
        + 2.0 * (fstar[right, 1] + fstar[right, 5] + fstar[right, 8])
    )
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

    i_top = top % nx
    x_top = X_lat[i_top]
    y_top = np.full_like(i_top, Y_lat[ny - 1], dtype=float)
    ux_t, uy_t = tg_analytical_velocity_field(x_top, y_top, t_physical, cfg.tg_u0, L_lat, nu)
    U[top, 0] = ux_t
    U[top, 1] = uy_t
    ro[top] = (1.0 / (1.0 + U[top, 1])) * (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    )
    fstar[top, 4] = fstar[top, 2] - (2.0 / 3.0) * ro[top] * U[top, 1]
    fstar[top, 7] = (
        fstar[top, 5]
        + 0.5 * (fstar[top, 1] - fstar[top, 3])
        - (1.0 / 6.0) * ro[top] * U[top, 1]
        - 0.5 * ro[top] * U[top, 0]
    )
    fstar[top, 8] = (
        fstar[top, 6]
        - 0.5 * (fstar[top, 1] - fstar[top, 3])
        - (1.0 / 6.0) * ro[top] * U[top, 1]
        + 0.5 * ro[top] * U[top, 0]
    )

    i_bottom = bottom % nx
    x_bottom = X_lat[i_bottom]
    y_bottom = np.full_like(i_bottom, Y_lat[0], dtype=float)
    ux_t, uy_t = tg_analytical_velocity_field(x_bottom, y_bottom, t_physical, cfg.tg_u0, L_lat, nu)
    U[bottom, 0] = ux_t
    U[bottom, 1] = uy_t
    ro[bottom] = (1.0 / (1.0 - U[bottom, 1])) * (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    )
    fstar[bottom, 2] = fstar[bottom, 4] + (2.0 / 3.0) * ro[bottom] * U[bottom, 1]
    fstar[bottom, 5] = (
        fstar[bottom, 7]
        - 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + (1.0 / 6.0) * ro[bottom] * U[bottom, 1]
        + 0.5 * ro[bottom] * U[bottom, 0]
    )
    fstar[bottom, 6] = (
        fstar[bottom, 8]
        + 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + (1.0 / 6.0) * ro[bottom] * U[bottom, 1]
        - 0.5 * ro[bottom] * U[bottom, 0]
    )

    corners = np.array([
        0,
        nx - 1,
        nx * (ny - 1),
        nx * ny - 1,
    ])
    corner_i = np.array([0, nx - 1, 0, nx - 1])
    corner_j = np.array([0, 0, ny - 1, ny - 1])
    x_corner = X_lat[corner_i]
    y_corner = Y_lat[corner_j]
    ux_c, uy_c = tg_analytical_velocity_field(x_corner, y_corner, t_physical, cfg.tg_u0, L_lat, nu)
    ro[corners] = 1.0
    U[corners, 0] = ux_c
    U[corners, 1] = uy_c
    fstar[corners] = compute_feq(ro[corners], U[corners], lattice)


class TGAnalyticalBC:
    name = "tg_analytical"

    def apply(self, fstar, ro, U, state, cfg, ttt: int) -> None:
        apply_bc_analytical(fstar, ro, U, ttt * state.dt, cfg, state.idx, state.lattice)


register_boundary(TGAnalyticalBC())
