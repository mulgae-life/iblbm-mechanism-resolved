"""Supplementary Table S10 — 생산 baseline 3건의 과도 특성 요약 재현 드라이버.

논문 Supplementary Material Sec. S4.2 Table S10과 본문 Appendix E의 값을,
생산 explicit-history baseline 이력(main Table IV와 동일 계보)에서 그대로 재계산한다.
추정기는 terminal_velocity.extract_terminal_velocity를 재사용한다
(rolling 표준편차 최소 plateau 창, 마지막 10% 표본 제외, t*_99 = 0.99·종단 최초 도달,
peak = 원시 이력 최댓값).

실행:
    python scripts/analysis/table_s10_production.py
"""

import json
from pathlib import Path

from terminal_velocity import extract_terminal_velocity

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_BASELINES = {
    "1.01": "data/single_particle_sedimentation/method_matrix/rho101/df_bgk_verlet_explicit_history_p4",
    "1.1": "data/single_particle_sedimentation/method_matrix/rho110/df_bgk_verlet_explicit_history_p4",
    "1.5": "data/single_particle_sedimentation/method_matrix/rho150/df_bgk_verlet_explicit_history",
}


def main() -> None:
    print("Table S10 — production explicit-history baseline (DF / Peskin 4-point / Velocity-Verlet)")
    header = f"{'rho_s/rho_f':>11} | {'term v_y*':>9} | {'peak v_y*':>9} | {'Delta%':>7} | {'t*_99':>6} | {'plateau std%':>12}"
    print(header)
    print("-" * len(header))
    for rho, subdir in PRODUCTION_BASELINES.items():
        path = REPO_ROOT / subdir / "sedimentation_history.json"
        with open(path) as f:
            history = json.load(f)
        r = extract_terminal_velocity(history)
        if r is None:
            raise RuntimeError(f"추출 실패: {path}")
        fluct_pct = r["vy_star_std"] / r["vy_star_mean"] * 100
        print(f"{rho:>11} | {r['vy_star_mean']:9.4f} | {r['vy_peak']:9.4f} | "
              f"{r['overshoot_pct']:7.3f} | {r['t_star_99']:6.2f} | {fluct_pct:12.3f}")


if __name__ == "__main__":
    main()
