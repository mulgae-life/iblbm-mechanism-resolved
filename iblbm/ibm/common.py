"""IBM 공통 인프라 — regularized delta kernel + dispatch registry.

IBM 전송 연산 (Peskin 2002 계보)
  - 보간  u(X_l) = Σ_x u(x) δ_h(x − X_l) dx²
  - 분산  f(x)   = Σ_l F(X_l) δ_h(x − X_l) Δs
  - δ_h(x,y) = (1/dx²) φ(x/dx) φ(y/dx)   separable 2D kernel

등록 커널
  - `peskin4pt`  support 4dx, stencil radius a = 2  (5 × 5 = 25 grid pts)
  - `hat`        support 2dx, stencil radius a = 1  (3 × 3 = 9 grid pts)

Peskin 4-point stencil (a = 2, 5 × 5 지지)

       j
       ↑
   +2  · · · · ·
   +1  · · · · ·
    0  · · ■ · ·      ■ = 가장 가까운 격자점 (X_l 반올림)
   −1  · · · · ·      · = φ 지지 격자점
   −2  · · · · ·
        −2 −1  0 +1 +2  → i

Hat stencil (a = 1, 3 × 3 지지)

       j
       ↑
   +1  · · ·
    0  · ■ ·
   −1  · · ·
        −1  0 +1  → i

φ 정의
  Peskin 4-point (Roma-Peskin type, Peskin 2002 Acta Numerica)
    φ(r) = ⅛(3 − 2|r| + √(1 + 4|r| − 4r²))       0 ≤ |r| ≤ 1
    φ(r) = ⅛(5 − 2|r| − √(−7 + 12|r| − 4r²))     1 ≤ |r| ≤ 2
    φ(r) = 0                                        |r| > 2
  Hat (linear, C⁰)
    φ(r) = max(1 − |r|, 0)

참조
  - Peskin (2002) *The Immersed Boundary Method*, Acta Numerica
  - Roma, Peskin, Berger (1999) — 4-point formula 계보
  - Zhang et al. (2020) Eq. (12) — separable 2D delta 형식
  - Wang, Fan, Luo (2008) Eq. (14)-(15) — IB-LBM 4-point delta 인용
"""

from __future__ import annotations

from ..backend import xp as np


def delta_hat(r: np.ndarray) -> np.ndarray:
    """Hat (piecewise-linear) delta kernel, C⁰, support 2·dx.

    정의
        φ(r) = max(1 − |r|, 0)

    특성
      - stencil radius a = 1  (3 × 3 지지)
      - 최소 비용 bilinear-interpolation 계열 kernel
      - 0th moment  Σ_i φ(i − x) = 1  (임의 실수 x)

    Args
        r: dx로 정규화된 거리 배열 (r = (x − X_l) / dx)

    Returns
        φ(r) 배열
    """
    return np.maximum(1.0 - np.abs(r), 0.0)


def delta_peskin4pt(r: np.ndarray) -> np.ndarray:
    """Peskin 4-point regularized delta kernel, support 4·dx.

    정의 (Peskin 2002 Acta Numerica)
        φ(r) = ⅛(3 − 2|r| + √(1 + 4|r| − 4r²))       0 ≤ |r| ≤ 1
        φ(r) = ⅛(5 − 2|r| − √(−7 + 12|r| − 4r²))     1 ≤ |r| ≤ 2
        φ(r) = 0                                       |r| > 2

    특성 (IBM kernel 표준)
      - stencil radius a = 2  (5 × 5 지지)
      - 0th moment  Σ_i φ(i − x)        = 1    (Uhlmann 2005 Eq. (10a))
      - 1st moment  Σ_i (i − x) φ(i − x) = 0    (Uhlmann 2005 Eq. (10b))
      - C¹ 연속, 수렴 특성은 bilinear (hat) 대비 개선

    수치 안정성
        `max(..., 0)`로 음수 방지 (반올림 오차 대응)

    Args
        r: dx로 정규화된 거리 배열 (r = (x − X_l) / dx)

    Returns
        φ(r) 배열
    """
    ar = np.abs(r)
    result = np.zeros_like(ar)

    mask1 = ar <= 1.0
    mask2 = (~mask1) & (ar <= 2.0)

    r1 = ar[mask1]
    result[mask1] = (1.0 / 8.0) * (
        3.0 - 2.0 * r1 + np.sqrt(np.maximum(1.0 + 4.0 * r1 - 4.0 * r1**2, 0.0))
    )

    r2 = ar[mask2]
    result[mask2] = (1.0 / 8.0) * (
        5.0 - 2.0 * r2 - np.sqrt(np.maximum(-7.0 + 12.0 * r2 - 4.0 * r2**2, 0.0))
    )

    return result


# 하위 호환 alias
delta_function = delta_peskin4pt

# 디스패치 레지스트리: (함수, stencil 반경)
_DELTA_REGISTRY = {
    "peskin4pt": (delta_peskin4pt, 2),  # 5점 스텐실
    "hat":       (delta_hat, 1),         # 3점 스텐실
}


def get_delta(delta_type: str):
    """Delta kernel dispatch.

    Args
        delta_type: `"peskin4pt"` | `"hat"`

    Returns
        (delta_func, stencil_radius)
          - delta_func      : φ(r) 벡터화 함수
          - stencil_radius  : 한쪽 방향 격자 offset 수 (5-pt kernel → 2, 3-pt → 1)
    """
    return _DELTA_REGISTRY[delta_type]
