"""4-face 압력형 개방 경계 (모든 face에서 ρ 고정).

Face 구성
  - 모든 face에서 ρ = ρ_b 지정
  - face 법선 속도는 known population으로 복원 (Zou-He 반사형)
  - 접선 속도는 내부 2-점 외삽 (프로젝트 구현)

D2Q9 방향 + face별 미지 population
  - left face  (법선 방향 +x, 유체→내부)
        known   = {f_0, f_2, f_3, f_4, f_6, f_7}
        unknown = {f_1, f_5, f_8}
  - right face (법선 방향 −x)
        known   = {f_0, f_2, f_4, f_1, f_5, f_8}
        unknown = {f_3, f_6, f_7}
  - top face   (법선 방향 −y)
        known   = {f_0, f_1, f_3, f_2, f_5, f_6}
        unknown = {f_4, f_7, f_8}
  - bottom face (법선 방향 +y)
        known   = {f_0, f_1, f_3, f_4, f_7, f_8}
        unknown = {f_2, f_5, f_6}

Face 법선 속도 복원 (Zou-He Eq. (22) 회전/반사)
  - left    u_x =  1 − [f_0 + f_2 + f_4 + 2(f_3 + f_6 + f_7)] / ρ_b
  - right   u_x = −1 + [f_0 + f_2 + f_4 + 2(f_1 + f_5 + f_8)] / ρ_b
  - top     u_y = −1 + [f_0 + f_1 + f_3 + 2(f_2 + f_5 + f_6)] / ρ_b
  - bottom  u_y =  1 − [f_0 + f_1 + f_3 + 2(f_4 + f_7 + f_8)] / ρ_b

계보
  - 직선 face closure: Zou & He (1997) §3 Eq. (22-23)

프로젝트 구현
  - 4-face 동시 압력형 조합
  - 접선 성분 2차 내부 외삽
  - corner 4개에 대한 대각 내부 노드 U 복사 + equilibrium refill
"""

from __future__ import annotations

from ..backend import xp as np
from ..lbm import compute_feq
from .base import BoundaryIndices, register_boundary


def apply_bc_open_boundary(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    dens: float, idx: BoundaryIndices,
    lattice=None,
) -> None:
    """4-face 압력 개방 경계 적용 (ρ = dens 고정).

    단계
      1. face별 ρ ← dens
      2. face 법선 속도를 known population으로 복원 (Zou-He Eq. (22) 회전)
      3. 접선 속도를 내부 2-점 외삽
      4. face 미지 {f_i}를 Zou-He Eq. (23) 반사형으로 복원
      5. (lattice != None) corner 4개에 대각 내부 노드 U 복사 + `compute_feq` refill

    참조
      - Zou & He (1997) §3 — face closure
      - 접선 외삽 + corner refill은 프로젝트 구현
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx = idx.nx

    ro[left] = dens
    U[left, 0] = 1.0 - (
        fstar[left, 0] + fstar[left, 2] + fstar[left, 4]
        + 2.0 * (fstar[left, 3] + fstar[left, 7] + fstar[left, 6])
    ) / ro[left]
    U[left, 1] = (4.0 * U[left + 1, 1] - U[left + 2, 1]) / 3.0
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

    ro[top] = dens
    U[top, 1] = -1.0 + (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    ) / ro[top]
    U[top, 0] = (4.0 * U[top - nx, 0] - U[top - 2 * nx, 0]) / 3.0
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

    ro[bottom] = dens
    U[bottom, 1] = 1.0 - (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    ) / ro[bottom]
    U[bottom, 0] = (4.0 * U[bottom + nx, 0] - U[bottom + 2 * nx, 0]) / 3.0
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

    if lattice is not None:
        ny = idx.ny
        corners = np.array([
            0,
            nx - 1,
            nx * (ny - 1),
            nx * ny - 1,
        ])
        diag1 = np.array([
            nx + 1,
            2 * nx - 2,
            nx * (ny - 2) + 1,
            nx * (ny - 1) - 2,
        ])
        ro[corners] = dens
        U[corners] = U[diag1]
        fstar[corners] = compute_feq(ro[corners], U[corners], lattice)


class OpenBoundaryBC:
    name = "open_boundary"

    def apply(self, fstar, ro, U, state, cfg, ttt: int) -> None:
        apply_bc_open_boundary(fstar, ro, U, state.dens, state.idx, state.lattice)


register_boundary(OpenBoundaryBC())
