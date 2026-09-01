"""단일 입자 침강 이력에서 종단 속도와 과도 지표를 추출하는 추정기.

`table_s10_production.py` 가 보충자료 Table S10 을 재현할 때 이 함수를 쓴다.
plateau 구간은 rolling std 최소 위치로 자동 탐색하고, 벽면 감속이 섞이지 않도록
마지막 10% 를 탐색에서 제외한다.
"""

import numpy as np


def extract_terminal_velocity(history, min_points=20):
    """종단 속도 추출 — plateau 자동 탐색 (rolling std 기반)."""
    if not history or len(history) < min_points:
        return None

    vy_all = np.array([r["vy_star"] for r in history])
    vx_all = np.array([r["vx_star"] for r in history])

    if not np.all(np.isfinite(vy_all)):
        return None

    n = len(vy_all)

    # Rolling std로 가장 안정적인 구간 탐색
    window = max(n // 10, 5)
    rolling_std = np.full(n, np.inf)
    for i in range(window, n - window):
        seg = vy_all[i - window:i + window]
        rolling_std[i] = np.std(seg)

    # 벽면 감속 제거: 마지막 10% 제외
    cutoff = int(n * 0.9)
    search_region = rolling_std[:cutoff]
    valid = np.isfinite(search_region)

    if not np.any(valid):
        start, end = int(n * 0.5), int(n * 0.8)
    else:
        best_center = np.argmin(search_region)
        start = max(0, best_center - window)
        end = min(cutoff, best_center + window)

    if end - start < min_points:
        start = max(0, int(n * 0.5))
        end = min(n, int(n * 0.85))

    plateau_vy = vy_all[start:end]
    plateau_vx = vx_all[start:end]

    vy_terminal = float(np.mean(plateau_vy))
    t_all = np.array([r["t_star"] for r in history])

    # t*_99: 종단 속도 99% 도달 시간
    threshold_99 = 0.99 * vy_terminal
    t99_idx = np.where(vy_all >= threshold_99)[0]
    t_star_99 = float(t_all[t99_idx[0]]) if len(t99_idx) > 0 else float(t_all[-1])

    # 오버슈트
    vy_peak = float(np.max(vy_all))
    peak_idx = int(np.argmax(vy_all))
    t_star_peak = float(t_all[peak_idx])
    overshoot_pct = (vy_peak - vy_terminal) / vy_terminal * 100

    # 횡방향 드리프트
    vx_max = float(np.max(np.abs(vx_all)))
    x_all = np.array([r["x"] for r in history])
    x_drift = float(np.abs(x_all[-1] - x_all[0]))

    # 벽면 감속
    tail_start = int(n * 0.95)
    if tail_start < n - 2:
        wall_decel = float((vy_all[tail_start] - vy_all[-1]) / vy_all[tail_start] * 100)
    else:
        wall_decel = 0.0

    return {
        "vy_star_mean": vy_terminal,
        "vy_star_std": float(np.std(plateau_vy)),
        "vx_star_mean": float(np.mean(np.abs(plateau_vx))),
        "t_star_final": history[-1]["t_star"],
        "y_star_final": history[-1]["y_star"],
        "plateau_range": f"{start/n*100:.0f}-{end/n*100:.0f}%",
        "n_points": len(plateau_vy),
        "t_star_99": t_star_99,
        "vy_peak": vy_peak,
        "t_star_peak": t_star_peak,
        "overshoot_pct": overshoot_pct,
        "vx_max": vx_max,
        "x_drift": x_drift,
        "wall_decel_pct": wall_decel,
    }
