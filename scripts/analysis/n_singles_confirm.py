"""N-3·N-5·N-8 (+N-2 완주 후) 확인 산출 — 잠긴 규격의 기계 적용.

- N-3: marker retraction {0.0, 0.3, 0.5, 1.0}Δx — A-9 |v_y| 규격(std y*≤15.5)
  peak Re_f, 자기 기준 δ%(RET00 대비), G-1 게이트(RET00 vs 기존 baseline
  <0.01%), 단조성 판정.
- N-5: rho150 {MDF, DFC} none — Table B.5 정본 스프레드: Re_p(particle basis
  = Re_f×1.5), spread% = |Re_p(eh) − Re_p(none)| / 503.38 (Wang same-source
  분모, B.5 실측 1.27% 선례와 동일 규약), ≤1.3% 판정.
- N-8: extended_80D two-particle none — 단일-arm(light) peak-completion 분류
  전용. 창 y*≤71.5, dual-ε judge. paired 대비 금지.
- N-2: tall60D df none — eh(기존)와 스프레드(std 15.5·t60 51.5 두 창 보고).
  런 완주 후 --with-n2로 산출.

A-9 admissible·peak 함수를 임포트 재사용(규격 단일 구현 유지).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

import a9_table_c1_recompute as a9  # noqa: E402 — admissible·peak 규격 단일 구현
import dual_epsilon_consensus as dec  # noqa: E402
import build_immutable_table as bit  # noqa: E402 — series·창 규격 재사용 (N-8)

DATA = ROOT / "data"
OUT_JSON = HERE / "n_singles_result.json"

# Table B.5 정본 상수 (본 연구 실측 — same-source Wang 분모)
WANG_SAME_SOURCE_REP = 503.38
SPREAD_BAND_PCT = 1.3
RHO150 = 1.5  # Re_p = Re_f × rho_ratio

# A-9 잠긴 기준 (기존 baseline — G-1 게이트 대조값)
BASELINE_STD_RE_F_VY = 336.30302636503865  # a9_table_c1_result.json DF/BGK std

SP = DATA / "single_particle_sedimentation"
N3_DIRS = {
    "ret00": SP / "marker_offset/rho150_df_bgk_verlet_eh_ret00",
    "ret03": SP / "marker_offset/rho150_df_bgk_verlet_eh_ret03",
    "ret05": SP / "marker_offset/rho150_df_bgk_verlet_eh_ret05",
    "ret10": SP / "marker_offset/rho150_df_bgk_verlet_eh_ret10",
}
N5_PAIRS = {
    "MDF": (SP / "method_matrix/rho150/mdf_bgk_verlet_explicit_history_p4",
            SP / "method_matrix/rho150/mdf_bgk_verlet_none"),
    "DFC": (SP / "method_matrix/rho150/dfc_bgk_verlet_explicit_history_p4",
            SP / "method_matrix/rho150/dfc_bgk_verlet_none"),
}
N8_DIR = DATA / "two_particle_sedimentation/extended_80D/df_bgk_verlet_none_80D"
N2_EH_DIR = DATA / "single_particle_sedimentation_tall60D/method_matrix/rho150/df_bgk_verlet_explicit_history"
N2_NONE_DIR = DATA / "single_particle_sedimentation_tall60D/method_matrix/rho150/df_bgk_verlet_none"


def single_peak(run_dir: Path, cfg, limit_ystar: float) -> dict:
    recs, status = a9.load_history(run_dir)
    kept, stats = a9.admissible_records(recs, limit_ystar, cfg.xmax, cfg.ymax)
    if not kept:
        raise AssertionError(f"missing_samples: {run_dir}")
    pm = a9.peak_metrics(kept, cfg.Re)
    return {"dir": str(run_dir.relative_to(DATA)), "admissible": stats,
            "completed": bool(status.get("completed")),
            "termination_reason": status.get("termination_reason"), **pm}


def main() -> int:
    with_n2 = "--with-n2" in sys.argv
    cfg_std = a9.build_cfg(None)
    out = {"analysis": "N-3/N-5/N-8 (+N-2) locked confirmation",
           "spec": {"metric": "A-9 unsmoothed max |v_y|*/u_g × cfg.Re",
                    "windows": {"std": 15.5, "t60": [15.5, 51.5], "t80_two_particle": 71.5},
                    "spread_denominator": WANG_SAME_SOURCE_REP,
                    "spread_band_pct": SPREAD_BAND_PCT,
                    "baseline_std_Re_f_vy": BASELINE_STD_RE_F_VY}}

    # ---------------- N-3 — retraction 사다리 (자기 기준 단조성)
    n3 = {}
    for tag, d in N3_DIRS.items():
        n3[tag] = single_peak(d, cfg_std, 15.5)
    g1_dev_pct = a9.pct(BASELINE_STD_RE_F_VY, n3["ret00"]["Re_f_vy"])
    g1_pass = abs(g1_dev_pct) < 0.01
    base = n3["ret00"]["Re_f_vy"]
    deltas = {t: a9.pct(base, n3[t]["Re_f_vy"]) for t in ("ret03", "ret05", "ret10")}
    seq = [deltas["ret03"], deltas["ret05"], deltas["ret10"]]
    monotone_increasing = seq[0] <= seq[1] <= seq[2]
    monotone_decreasing = seq[0] >= seq[1] >= seq[2]
    out["n3"] = {
        "arms": n3,
        "g1_gate": {"dev_pct_vs_locked_baseline": g1_dev_pct, "pass": g1_pass},
        "delta_pct_vs_ret00": deltas,
        "monotone": monotone_increasing or monotone_decreasing,
        "direction": ("increasing" if monotone_increasing else
                      "decreasing" if monotone_decreasing else "non_monotone"),
        "abs_delta_Re_f": {t: n3[t]["Re_f_vy"] - base for t in ("ret03", "ret05", "ret10")},
    }

    # ---------------- N-5 — cross-IBM ablation 스프레드 (Table B.5 규약)
    n5 = {}
    for method, (eh_dir, none_dir) in N5_PAIRS.items():
        eh = single_peak(eh_dir, cfg_std, 15.5)
        no = single_peak(none_dir, cfg_std, 15.5)
        rep_eh, rep_no = eh["Re_f_vy"] * RHO150, no["Re_f_vy"] * RHO150
        spread_pct = abs(rep_eh - rep_no) / WANG_SAME_SOURCE_REP * 100.0
        n5[method] = {"eh": eh, "none": no,
                      "Re_p_eh": rep_eh, "Re_p_none": rep_no,
                      "spread_abs_Re_p": abs(rep_eh - rep_no),
                      "spread_pct_wang_denom": spread_pct,
                      "within_band": spread_pct <= SPREAD_BAND_PCT}
    out["n5"] = n5

    # ---------------- N-8 — 80D 단일-arm peak-completion 분류 (paired 금지)
    recs8 = bit.load_history(N8_DIR)
    st8 = json.loads((N8_DIR / "status.json").read_text())
    _case, exp2 = bit.expected_materialized_config("two", "TWO_PARTICLE_EXT80D_NONE")
    d_dom = float(exp2.cylinder_D_ratio)
    s8 = bit.series_two_particle(recs8, 1, 71.5, d_dom)
    j8 = dec.judge(bit.normalized_pct(s8))
    out["n8"] = {
        "dir": str(N8_DIR.relative_to(DATA)),
        "completed": bool(st8.get("completed")),
        "termination_reason": st8.get("termination_reason"),
        "window": {"ystar_limit": 71.5,
                   "end_reason": s8["window_end_reason"],
                   "end_t_star": s8["window_end_t_star"],
                   "n_admissible": s8["n_admissible"]},
        "judgment": j8,
        "paired_comparison": "forbidden_by_manifest(단일-arm 진단 전용)",
    }

    # ---------------- N-2 — tall60D eh vs none 스프레드 (완주 후)
    if with_n2:
        cfg_t60 = a9.build_cfg("tall60D")
        n2 = {}
        for win_tag, limit in (("std_15p5", 15.5), ("t60_51p5", 51.5)):
            eh = single_peak(N2_EH_DIR, cfg_t60, limit)
            no = single_peak(N2_NONE_DIR, cfg_t60, limit)
            rep_eh, rep_no = eh["Re_f_vy"] * RHO150, no["Re_f_vy"] * RHO150
            spread_pct = abs(rep_eh - rep_no) / WANG_SAME_SOURCE_REP * 100.0
            n2[win_tag] = {"eh": eh, "none": no,
                           "Re_p_eh": rep_eh, "Re_p_none": rep_no,
                           "spread_pct_wang_denom": spread_pct,
                           "within_band": spread_pct <= SPREAD_BAND_PCT}
        out["n2"] = n2

    # numpy scalar → 파이썬 스칼라 (JSON 직렬화를 위한 값 불변 캐스팅)
    OUT_JSON.write_text(json.dumps(
        out, indent=1,
        default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"WROTE {OUT_JSON}")
    # 핵심 요약 (수치는 판정 공개 단계)
    print("N-3 G-1:", "PASS" if g1_pass else "FAIL", f"(dev {g1_dev_pct:+.4f}%)")
    print("N-3 δ% (0.3/0.5/1.0Δx):",
          ", ".join(f"{deltas[t]:+.3f}%" for t in ("ret03", "ret05", "ret10")),
          "| direction:", out["n3"]["direction"])
    for m in n5:
        print(f"N-5 {m}: spread {n5[m]['spread_pct_wang_denom']:.3f}%",
              "(within band)" if n5[m]["within_band"] else "(EXCEEDS band)")
    print("N-8:", j8.get("completion_status"), "/", j8.get("observable_status"),
          "| window end:", out["n8"]["window"]["end_reason"])
    if with_n2:
        for w in out["n2"]:
            print(f"N-2 {w}: spread {out['n2'][w]['spread_pct_wang_denom']:.3f}%",
                  "(within band)" if out["n2"][w]["within_band"] else "(EXCEEDS band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
