"""진단 핵심 지표 (Cd/Cl, St, L₂ 오차, Taylor-Green 해석해, tail window 통계).

핵심 식
  - 항력/양력 계수 (원형 실린더 반지름 기준)
        C_D = F_x / (ρ U_ref² R)
        C_L = F_y / (ρ U_ref² R)
  - Strouhal 수
        St = f_shed D / U_ref
  - L₂ 상대 오차
        ε_L2 = ||u_num − u_ana||₂ / ||u_ana||₂
  - Taylor-Green 감쇠 와류 (도메인 [−L, L]²)
        u_x = −u_0 cos(π x/L) sin(π y/L) exp(−2ν(π/L)² t)
        u_y =  u_0 sin(π x/L) cos(π y/L) exp(−2ν(π/L)² t)
  - 재순환 길이
        L_r = L / D   (centerline의 u_x 부호 전환점에서 실린더 후면까지)

Tail window 통계
  - `STEADY_TAIL_FRAC = 0.3`  후반부 tail 비율 (수렴 검정 + 진폭 집계 공용)

용도
  - benchmark 후처리 (Cd/Cl 평균, St, L_r)
  - runtime logging (L₂ 수렴 판정)
"""
from __future__ import annotations

import numpy as np

STEADY_TAIL_FRAC = 0.3


def tail_start_index(length: int, frac: float = STEADY_TAIL_FRAC) -> int:
    """후반 tail 구간 시작 인덱스 (길이 N, 비율 frac → 시작 = ⌊N(1−frac)⌋)."""
    if length <= 0:
        raise ValueError("empty history array")
    return min(max(int(length * (1.0 - frac)), 0), length - 1)

def tail_values(values, frac: float = STEADY_TAIL_FRAC) -> np.ndarray:
    """후반 tail 구간 슬라이스 (float64 ndarray)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty history array")
    return arr[tail_start_index(arr.size, frac):]

def tail_mean(values, frac: float = STEADY_TAIL_FRAC) -> float:
    """후반 tail 구간 산술 평균 (Cd 평균 집계용)."""
    return float(np.mean(tail_values(values, frac)))

def tail_peak_to_peak_amp(values, frac: float = STEADY_TAIL_FRAC) -> float:
    """후반 tail 구간 반-진폭 ½(max − min) (signed 시계열의 진동 진폭용)."""
    tail = tail_values(values, frac)
    return float((tail.max() - tail.min()) / 2.0)

def compute_cd_cl(
    fib: np.ndarray, ro: np.ndarray,
    u_ref: float, r_lattice: float,
) -> tuple[float, float]:
    """Eulerian IB 힘장 합으로 Cd/Cl 산출 (DF/MDF용).

    식
      - C_D = F_x / (ρ̄ U_ref² R)
      - C_L = F_y / (ρ̄ U_ref² R)
      - ρ̄ = mean(ro)

    부호 규약
      Cd = |Σ fib[:,0]| / denom     (양수 강제)
      Cl = Σ fib[:,1] / denom       (부호 유지 → 와류 방출 교대 보존)

    Args
      - fib          (nodenums, 2) IB 체적력 (Eulerian spread)
      - ro           (nodenums,)   밀도장
      - u_ref        참조 속도 (lattice 단위)
      - r_lattice    실린더 반지름 (lattice 단위)

    Returns
      - (Cd, Cl) — float
    """
    denom = np.mean(ro) * u_ref**2 * r_lattice
    # Cd = abs(sum(fib[:,0])) / denom
    Cd = np.abs(np.sum(fib[:, 0])) / denom
    Cl = np.sum(fib[:, 1]) / denom  # 부호 유지 (와류 방출 패턴 보존)
    return Cd, Cl

def compute_cd_cl_dfc(
    dfc_force_lagr: np.ndarray, ro_mean: float,
    u_ref: float, r_lattice: float,
) -> tuple[float, float]:
    """DFC 마커별 힘 집계량 기반 Cd/Cl 산출.

    `compute_cd_cl()` 재사용 불가 사유
      - DF : `fib` = Eulerian spread → Σ fib = Σ_k F_lagr(k) Δs
      - DFC: 마커 aggregate에 이미 `Δs` 포함 → 마커 합 그대로 사용
      - DFC를 spread 후 sum하면 Δs가 이중 적용 (오류)

    식
      - C_D = F_x / (ρ̄ U_ref² R)
      - C_L = F_y / (ρ̄ U_ref² R)

    Args
      - dfc_force_lagr  (Lb, 2) DFC 마커 힘 집계 (Δs 재곱 금지)
      - ro_mean         평균 밀도
      - u_ref           참조 속도 (lattice 단위)
      - r_lattice       실린더 반지름 (lattice 단위)

    Returns
      - (Cd, Cl) — float
    """
    total_force = np.sum(dfc_force_lagr, axis=0)  # (2,)
    denom = ro_mean * u_ref**2 * r_lattice

    Cd = float(np.abs(total_force[0])) / denom
    Cl = float(total_force[1]) / denom

    return Cd, Cl

def check_convergence(
    Eux_new: np.ndarray, Euy_new: np.ndarray,
    Eux_old: np.ndarray, Euy_old: np.ndarray,
) -> float:
    """2-스텝 속도장 L₂ 상대 오차 (수렴 판정용).

    식
      - ε = √(Σ((u_new − u_old)²) / Σ(u_old²))
      - 분모 0이면 1.0 반환 (초기 수렴 미달 처리)

    Args
      - Eux_new, Euy_new   현재 스텝 2D 속도장
      - Eux_old, Euy_old   이전 체크 시점 2D 속도장

    Returns
      - L₂ 상대 오차 (무차원 scalar)
    """
    numer = np.sum((Eux_new - Eux_old)**2 + (Euy_new - Euy_old)**2)
    denom = np.sum(Eux_old**2 + Euy_old**2)
    if denom == 0.0:
        return 1.0
    return np.sqrt(numer / denom)

def compute_strouhal(
    Cl_history: np.ndarray,
    steps: np.ndarray,
    D_lattice: float,
    u_ref: float,
    check_interval: int,
) -> float:
    """Cl 시계열로부터 Strouhal 수 추정.

    식
      - St = f_shed D / U_ref

    알고리즘
      1. 후반 50% Cl 시계열 추출
      2. 평균 제거 (디트렌드)
      3. `np.fft.rfft` 파워 스펙트럼
      4. DC 제외 최대 파워 피크 주파수 f_peak
      5. `St = f_shed D_lattice / u_ref`

    |Cl| rectified 입력 처리 (signed Cl 판별 + aliasing 대응)
      - 배경: `compute_cd_cl()`이 정상류 케이스에 `np.abs()` 적용 → Cl_history ≥ 0 가능
      - |Cl|의 FFT 피크 주파수 = 2 · f_shed (rectified)
      - 2 · f_shed > Nyquist → aliasing 발생 가능
      - 두 해석 병행 후 자기일관성으로 판별
          · 비앨리어싱: f_shed = f_peak / 2
          · 앨리어싱  : f_shed = (f_sampling − f_peak) / 2
      - 임계: St_thresh = D / (4 · check_interval · u_ref)
          · St_direct ≥ 0.1 (와류 방출 물리 범위) → 비앨리어싱 채택

    프로젝트 규약
      - 후반 50% window + |Cl| rectified 판별은 현재 benchmark 후처리 규칙

    Args
      - Cl_history       Cl 시계열 (signed 또는 |Cl|)
      - steps            스텝 번호 배열 (check_interval 간격)
      - D_lattice        실린더 격자 직경 (= D_ratio × (NN−1))
      - u_ref            참조 속도 (lattice 단위, inflow_u)
      - check_interval   데이터 샘플링 간격 (스텝 수)

    Returns
      - Strouhal 수 (scalar)
    """
    Cl_arr = np.asarray(Cl_history, dtype=float)
    N = len(Cl_arr)
    half = Cl_arr[N // 2:]

    is_rectified = bool(np.all(Cl_arr >= -1e-10))

    half_detrended = half - np.mean(half)
    fft_vals = np.fft.rfft(half_detrended)
    power = np.abs(fft_vals) ** 2
    power[0] = 0  # DC 제거

    peak_idx = int(np.argmax(power))
    N_half = len(half)
    f_peak = peak_idx / (N_half * check_interval)

    if is_rectified:
        # |Cl| 데이터: FFT 피크는 2×f_shed(직접) 또는 앨리어싱 주파수
        # 두 해석 모두 계산 후 자기일관성으로 판별
        f_sampling = 1.0 / check_interval
        St_dealiased = ((f_sampling - f_peak) / 2) * D_lattice / u_ref
        St_direct = (f_peak / 2) * D_lattice / u_ref

        # 앨리어싱 임계: St_threshold = D / (4 × ci × u)
        # St > threshold → 2f_shed > f_Nyquist → 앨리어싱 발생
        # St_direct는 항상 < threshold, St_dealiased는 항상 ≥ threshold
        # St_direct가 와류 방출 물리 범위(≥ 0.1)이면 앨리어싱 미발생으로 판단
        if St_direct >= 0.1:
            f_shed = f_peak / 2
        else:
            f_shed = (f_sampling - f_peak) / 2
    else:
        # signed Cl → FFT 피크가 직접 f_shed
        f_shed = f_peak

    return f_shed * D_lattice / u_ref

def tg_analytical_velocity_field(
    X: np.ndarray, Y: np.ndarray, t: float,
    u0: float, L: float, nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Taylor-Green 감쇠 와류 해석해 속도장.

    식
      - u_x = −u_0 cos(π x/L) sin(π y/L) exp(−2ν(π/L)² t)
      - u_y =  u_0 sin(π x/L) cos(π y/L) exp(−2ν(π/L)² t)

    Args
      - X, Y   물리 좌표 배열 ([−L, L] 범위)
      - t      물리 시간
      - u0     초기 속도 진폭
      - L      도메인 반폭
      - nu     동점성 계수 ν

    Returns
      - (ux, uy) — 각 성분 ndarray
    """
    decay = np.exp(-2.0 * nu * (np.pi / L) ** 2 * t)
    ux = -u0 * np.cos(np.pi * X / L) * np.sin(np.pi * Y / L) * decay
    uy = u0 * np.sin(np.pi * X / L) * np.cos(np.pi * Y / L) * decay
    return ux, uy

def compute_l2_error(
    U_num: np.ndarray, U_ana: np.ndarray,
) -> float:
    """수치해 vs 해석해 L₂ 상대 오차.

    식
      - ε = ||U_num − U_ana||₂ / ||U_ana||₂
      - 분모 0이면 `inf` 반환

    Args
      - U_num   수치 속도 (N, 2) 또는 (ny, nx) 등
      - U_ana   해석해 속도 (같은 shape)

    Returns
      - L₂ 상대 오차 (무차원 float)
    """
    diff = U_num - U_ana
    numer = np.sqrt(np.sum(diff ** 2))
    denom = np.sqrt(np.sum(U_ana ** 2))
    if denom == 0.0:
        return float('inf')
    return float(numer / denom)

def compute_recirculation_length(
    Eux: np.ndarray,
    dx: float,
    dy: float,
    cx: float,
    cy: float,
    D: float,
) -> float:
    """실린더 하류 재순환 길이 L_r = L/D 산출.

    알고리즘
      - centerline (j = round(cy/dy)) 행의 u_x(i) 스캔
      - 실린더 후면 x_rear = cx + D/2 바로 뒤 (i_start = ⌈x_rear/dx⌉ + 1)부터 탐색
      - u_x < 0 → u_x ≥ 0 전환점을 선형 보간으로 x_cross 확정
      - L_r = (x_cross − x_rear) / D

    규약
      - 재순환 길이는 실린더 후면 기준 거리를 직경 D로 정규화해 보고한다.

    Args
      - Eux      2D x-속도장 (ny, nx)
      - dx, dy   격자 간격 (도메인 좌표)
      - cx, cy   실린더 중심 (도메인 좌표)
      - D        실린더 직경 (도메인 좌표)

    Returns
      - L_r = L/D (scalar), 전환점 없으면 NaN
    """
    ny, nx = Eux.shape
    j_center = int(round(cy / dy))
    ux_line = Eux[j_center, :]

    x_rear = cx + D / 2.0
    # 실린더 후면 바로 뒤부터 탐색 (1격자 여유)
    i_start = int(np.ceil(x_rear / dx)) + 1

    for i in range(i_start, nx - 1):
        if ux_line[i] < 0 and ux_line[i + 1] >= 0:
            # 선형 보간
            x0 = i * dx
            x1 = (i + 1) * dx
            u0 = ux_line[i]
            u1 = ux_line[i + 1]
            x_cross = x0 + (0.0 - u0) / (u1 - u0) * (x1 - x0)
            return (x_cross - x_rear) / D

    return float('nan')

def compute_cl_amplitude(Cl_history: np.ndarray) -> float:
    """Cl 시계열 후반 50% 진폭 집계.

    규칙 (signed vs rectified 자동 분기)
      - `all(Cl ≥ −1e-10)` → |Cl| rectified → amp = max(후반 50%)
      - 그 외 (signed Cl)  → amp = (max − min) / 2

    Args
      - Cl_history   Cl 시계열 (signed 또는 |Cl|)

    Returns
      - Cl 진폭 (float)
    """
    Cl_arr = np.asarray(Cl_history, dtype=float)
    half = Cl_arr[len(Cl_arr) // 2:]

    if np.all(Cl_arr >= -1e-10):
        # |Cl| 데이터: 진폭 = max(|Cl|)
        return float(np.max(half))
    else:
        # signed Cl: 진폭 = (max - min) / 2
        return float((np.max(half) - np.min(half)) / 2.0)
