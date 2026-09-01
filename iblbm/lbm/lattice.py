"""D2Q9 격자 상수.

방향 인덱스 (e_i 화살표 배치)

        6   2   5
         ╲  │  ╱
      3 ─── 0 ─── 1
         ╱  │  ╲
        7   4   8

  - i = 0        정지,  |e₀| = 0,        w₀ = 4/9
  - i ∈ {1..4}   축,    |e_i| = c,       w_i = 1/9
  - i ∈ {5..8}   대각,  |e_i| = √2·c,    w_i = 1/36

단위
  - 격자 단위 c ≡ Δx/Δt = 1
  - 음속 제곱 c_s² = c²/3 = 1/3
  - 반대방향 opp[i]: e_{opp(i)} = −e_i
      → bounce-back, TRT의 대칭/반대칭 분해 (f⁺, f⁻) 에 사용
"""

from dataclasses import dataclass

from ..backend import xp as np


@dataclass(frozen=True)
class D2Q9:
    # (9, 2) 격자 속도 e_i
    e: np.ndarray
    # (9,)   가중치 w_i
    w: np.ndarray
    # (9,)   반대방향 인덱스 opp[i] (e_{opp(i)} = −e_i)
    opp: np.ndarray
    # 음속 제곱 c_s²
    cs2: float = 1 / 3

    def __hash__(self):
        return id(self)


def make_d2q9() -> D2Q9:
    e = np.array([
        [0, 0],                            # 0: 정지
        [1, 0], [0, 1], [-1, 0], [0, -1],  # 1..4: 축
        [1, 1], [-1, 1], [-1, -1], [1, -1],# 5..8: 대각
    ], dtype=np.float64)

    w = np.array([
        4 / 9,
        1 / 9, 1 / 9, 1 / 9, 1 / 9,
        1 / 36, 1 / 36, 1 / 36, 1 / 36,
    ], dtype=np.float64)

    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
    return D2Q9(e=e, w=w, opp=opp)


def make_d2q9_mrt_matrices():
    """D2Q9 raw-moment 변환 행렬 M, M⁻¹.

    변환식
        m = M · f,    f = M⁻¹ · m

    모멘트 순서 (Lallemand & Luo 2000, NASA/CR-2000-210103, Eq. 3.4)
        m = (ρ, e, ε, j_x, q_x, j_y, q_y, p_xx, p_xy)ᵀ

        - ρ, j_x, j_y      보존 모멘트 (충돌 불변)
        - e                운동 에너지
        - ε                에너지 제곱
        - q_x, q_y         에너지 flux
        - p_xx, p_xy       점성 응력 성분

    행렬 M (Lallemand-Luo Eq. 3.6): 열은 i=0..8 격자 방향, 행은 위 모멘트 순서

    Returns
        (M, M⁻¹), shape (9, 9), numpy float64 쌍
    """
    # MRT 행렬은 backend 무관: 항상 numpy
    import numpy as _np

    M = _np.array([
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1],  # ρ
        [-4, -1, -1, -1, -1,  2,  2,  2,  2],  # e
        [ 4, -2, -2, -2, -2,  1,  1,  1,  1],  # ε
        [ 0,  1,  0, -1,  0,  1, -1, -1,  1],  # j_x
        [ 0, -2,  0,  2,  0,  1, -1, -1,  1],  # q_x
        [ 0,  0,  1,  0, -1,  1,  1, -1, -1],  # j_y
        [ 0,  0, -2,  0,  2,  1,  1, -1, -1],  # q_y
        [ 0,  1, -1,  1, -1,  0,  0,  0,  0],  # p_xx
        [ 0,  0,  0,  0,  0,  1, -1,  1, -1],  # p_xy
    ], dtype=_np.float64)

    M_inv = _np.linalg.inv(M)
    return M, M_inv
