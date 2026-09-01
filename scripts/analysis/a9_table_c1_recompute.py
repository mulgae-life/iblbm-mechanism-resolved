"""A-9. Table C.1 재구성 공식 후처리 — locked confirmation.

A-9 규격의 공식 산출 스크립트다.
탐색 단계 표의 값이 이미 공개된 상태이므로 blind 예측이 아니라
**locked confirmation**으로 분류한다 — 이 스크립트는 실행 전 해시로 고정한다.

규격 (사전 고정):
- 주 지표: |v_y| 피크 (본 논문 Table IV 관측량 계약 v_y* = |v_y|_max/u_g).
  속도 크기 sqrt(vx²+vy²)와의 차이는 민감도 각주용으로 병기한다.
- Re 환산: Re_f = vy_star_max × cfg.Re. cfg.Re(= u_g·D/ν, 참조 Re)는 하드코딩하지
  않고 생산 builder(make_sedimentation_config)를 각 케이스 인자로 호출해 얻는다.
  vy_star는 각 런이 자기 u_g로 정규화한 값이므로 이 환산은 격자 해상도 불변이다.
- admissible record 규격 (첫 contact crossing 전만 사용):
  * 관측 한계: standard y*≤15.5 / tall60D y*≤51.5(전 지평)·y*≤15.5(정합 창) /
    tall80D y*≤71.5. 한계 초과 첫 레코드에서 절단(절단은 위반이 아님).
  * 유한값: 레코드의 모든 수치 필드가 finite.
  * 도메인 내부 중심: 0 < x < xmax, 0 < y < ymax (물리 좌표, cfg에서 취득).
  * 단조 step: step이 직전 레코드보다 순증가.
  * 비정상 위치 점프: 인접 레코드 간 Δy* > 0.1 이면 위반.
    (물리 상한: |v_y*|≈1.3, 100스텝 간 Δt*≈0.023 → Δy*≈0.03. 0.1은 그 3배 이상.)
  * 위반 레코드는 제외하고 제외 사실·개수를 산출물에 기록한다.
- legacy final_step 오프셋(status.final_step = 마지막 history step − 1)은 기록만
  하고 물리 유효성 판정은 위 기하·접촉 조건으로만 한다.
- 표본 최대는 무평활(raw record max).

locked confirmation 대조 목표 (탐색 단계 정본, Re 소수 2자리):
  std@15.5 / t60@15.5 / t60 full, 채널·관측구간·합산 %, 80D 포화 +0.179/+0.176%.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scenarios.sedimentation import make_sedimentation_config  # noqa: E402

DATA = ROOT / "data"
OUT_JSON = Path(__file__).with_name("a9_table_c1_result.json")

JUMP_LIMIT_YSTAR = 0.1

# (구성 라벨, std 디렉터리, tall60 디렉터리) — 6구성. tall80은 DF 2구성만 실재.
CONFIGS = [
    ("DF/BGK",  "df_bgk_verlet_explicit_history",     "df_bgk_verlet_explicit_history"),
    ("DF/TRT",  "df_trt_verlet_explicit_history",     "df_trt_verlet_explicit_history"),
    ("MDF/BGK", "mdf_bgk_verlet_explicit_history_p4", "mdf_bgk_verlet_explicit_history"),
    ("MDF/TRT", "mdf_trt_verlet_explicit_history",    "mdf_trt_verlet_explicit_history"),
    ("DFC/BGK", "dfc_bgk_verlet_explicit_history_p4", "dfc_bgk_verlet_explicit_history"),
    ("DFC/TRT", "dfc_trt_verlet_explicit_history",    "dfc_trt_verlet_explicit_history"),
]
TALL80_CONFIGS = [
    ("DF/BGK", "df_bgk_verlet_explicit_history"),
    ("DF/TRT", "df_trt_verlet_explicit_history"),
]

# 탐색 단계 값 (대조 목표; Re 2자리 반올림 기준)
LOCKED_TARGETS = {
    "DF/BGK":  (336.30, 341.00, 349.02),
    "DF/TRT":  (336.54, 341.24, 349.28),
    "MDF/BGK": (339.18, 344.00, 354.75),
    "MDF/TRT": (339.66, 344.18, 354.99),
    "DFC/BGK": (337.64, 342.33, 353.26),
    "DFC/TRT": (337.66, 342.35, 353.24),
}
LOCKED_T80_SAT = {"DF/BGK": 0.179, "DF/TRT": 0.176}


def build_cfg(case_name: str | None):
    return make_sedimentation_config(
        rho_ratio=1.5, ibm_method="DF", delta_type="peskin4pt", collision_model="BGK",
        NN=1281, verbose=False, enable_rotation=True, time_integrator="verlet",
        incompressible_lbgk=False, settling_inertia_model="explicit_history",
        sedimentation_stop_at_contact=True, sedimentation_reference_basis="particle_basis",
        diagnostics_interval=0, mdf_iterations=1,
        **({"case_name": case_name} if case_name else {}),
    )


def load_history(run_dir: Path):
    with open(run_dir / "sedimentation_history.json") as f:
        recs = json.load(f)
    with open(run_dir / "status.json") as f:
        status = json.load(f)
    return recs, status


def admissible_records(recs, limit_ystar, xmax, ymax):
    """규격의 검사를 순서대로 적용. 반환: (admissible, 통계 dict)."""
    kept, prev_step, prev_ystar = [], None, None
    n_jump = n_nonfinite = n_domain = n_step = 0
    truncated_at = None
    for rec in recs:
        vals = [rec[k] for k in rec if isinstance(rec[k], (int, float))]
        if not all(math.isfinite(v) for v in vals):
            n_nonfinite += 1
            continue
        if prev_step is not None and rec["step"] <= prev_step:
            n_step += 1
            continue
        if not (0.0 < rec["x"] < xmax and 0.0 < rec["y"] < ymax):
            n_domain += 1
            continue
        if prev_ystar is not None and rec["y_star"] - prev_ystar > JUMP_LIMIT_YSTAR:
            n_jump += 1
            continue
        if rec["y_star"] > limit_ystar:
            truncated_at = rec["step"]
            break
        kept.append(rec)
        prev_step, prev_ystar = rec["step"], rec["y_star"]
    stats = {
        "n_input": len(recs), "n_admissible": len(kept),
        "excluded_nonfinite": n_nonfinite, "excluded_step_order": n_step,
        "excluded_domain": n_domain, "excluded_jump": n_jump,
        "truncated_at_step": truncated_at,
        "last_admissible_step": kept[-1]["step"] if kept else None,
    }
    return kept, stats


def peak_metrics(recs, ref_re):
    """무평활 표본 최대: 주 지표 |v_y|*, 각주용 속도 크기."""
    vy_max = max(r["vy_star"] for r in recs)
    vy_argstep = max(recs, key=lambda r: r["vy_star"])["step"]
    speed_max = max(math.hypot(r["vx_star"], r["vy_star"]) for r in recs)
    return {
        "vy_star_max": vy_max,
        "Re_f_vy": vy_max * ref_re,
        "peak_step": vy_argstep,
        "speed_star_max": speed_max,
        "Re_f_speed": speed_max * ref_re,
        "speed_vs_vy_pct": (speed_max - vy_max) / vy_max * 100.0,
    }


def pct(a, b):
    return (b - a) / a * 100.0


def main() -> int:
    cfg_std = build_cfg(None)
    cfg_t60 = build_cfg("tall60D")
    cfg_t80 = build_cfg("tall80D")
    ref_re = {"std": float(cfg_std.Re), "t60": float(cfg_t60.Re), "t80": float(cfg_t80.Re)}
    domains = {
        "std": (cfg_std.xmax, cfg_std.ymax),
        "t60": (cfg_t60.xmax, cfg_t60.ymax),
        "t80": (cfg_t80.xmax, cfg_t80.ymax),
    }
    print(f"참조 Re: std {ref_re['std']:.6f} / t60 {ref_re['t60']:.6f} / t80 {ref_re['t80']:.6f}")

    results, confirm_fail = {}, []
    for label, std_tag, t60_tag in CONFIGS:
        std_dir = DATA / "single_particle_sedimentation/method_matrix/rho150" / std_tag
        t60_dir = DATA / "single_particle_sedimentation_tall60D/method_matrix/rho150" / t60_tag
        entry = {}
        # standard 24D, y*≤15.5 (full contact-safe)
        recs, status = load_history(std_dir)
        adm, stats = admissible_records(recs, 15.5, *domains["std"])
        entry["std_15p5"] = {**peak_metrics(adm, ref_re["std"]), "admissible": stats,
                             "final_step_offset": status.get("final_step") is not None
                             and recs[-1]["step"] - status["final_step"]}
        # tall60D: matched 15.5 + full 51.5
        recs, status = load_history(t60_dir)
        for key, lim in (("t60_15p5", 15.5), ("t60_51p5", 51.5)):
            adm, stats = admissible_records(recs, lim, *domains["t60"])
            entry[key] = {**peak_metrics(adm, ref_re["t60"]), "admissible": stats,
                          "final_step_offset": status.get("final_step") is not None
                          and recs[-1]["step"] - status["final_step"]}
        re3 = (entry["std_15p5"]["Re_f_vy"], entry["t60_15p5"]["Re_f_vy"], entry["t60_51p5"]["Re_f_vy"])
        entry["decomposition_pct"] = {
            "channel": pct(re3[0], re3[1]),
            "horizon": pct(re3[1], re3[2]),
            "combined": pct(re3[0], re3[2]),
        }
        tgt = LOCKED_TARGETS[label]
        rounded = tuple(round(v, 2) for v in re3)
        entry["locked_confirmation"] = {"target": tgt, "recomputed": rounded,
                                        "match": rounded == tgt}
        if rounded != tgt:
            confirm_fail.append((label, tgt, rounded))
        results[label] = entry

    # tall80D 포화 (DF 2구성 한정)
    for label, t80_tag in TALL80_CONFIGS:
        t80_dir = DATA / "single_particle_sedimentation_tall80D/method_matrix/rho150" / t80_tag
        recs, status = load_history(t80_dir)
        adm, stats = admissible_records(recs, 71.5, *domains["t80"])
        m = peak_metrics(adm, ref_re["t80"])
        sat = pct(results[label]["t60_51p5"]["Re_f_vy"], m["Re_f_vy"])
        results[label]["t80_71p5"] = {**m, "admissible": stats,
                                      "saturation_60D_to_80D_pct": sat,
                                      "final_step_offset": status.get("final_step") is not None
                                      and recs[-1]["step"] - status["final_step"]}
        tgt = LOCKED_T80_SAT[label]
        ok = round(sat, 3) == tgt
        results[label]["t80_71p5"]["locked_confirmation"] = {"target_pct": tgt,
                                                            "recomputed_pct": round(sat, 3),
                                                            "match": ok}
        if not ok:
            confirm_fail.append((label + " t80sat", tgt, round(sat, 3)))

    # 표 출력
    print(f"\n{'구성':8s} {'std@15.5':>9s} {'t60@15.5':>9s} {'t60 full':>9s} "
          f"{'채널%':>7s} {'구간%':>7s} {'합산%':>7s} {'속도크기차%':>9s} {'확인':>4s}")
    for label, e in results.items():
        d = e["decomposition_pct"]
        fn_diff = max(e[k]["speed_vs_vy_pct"] for k in ("std_15p5", "t60_15p5", "t60_51p5"))
        print(f"{label:8s} {e['std_15p5']['Re_f_vy']:9.2f} {e['t60_15p5']['Re_f_vy']:9.2f} "
              f"{e['t60_51p5']['Re_f_vy']:9.2f} {d['channel']:+7.3f} {d['horizon']:+7.3f} "
              f"{d['combined']:+7.3f} {fn_diff:9.4f} "
              f"{'일치' if e['locked_confirmation']['match'] else '불일치'}")
    for label, _ in TALL80_CONFIGS:
        t = results[label]["t80_71p5"]
        print(f"{label:8s} 80D 포화 {t['saturation_60D_to_80D_pct']:+.3f}% "
              f"(목표 +{t['locked_confirmation']['target_pct']}%) "
              f"{'일치' if t['locked_confirmation']['match'] else '불일치'}")

    excl_total = sum(
        e[k]["admissible"][f] for e in results.values()
        for k in e if isinstance(e.get(k), dict) and "admissible" in e[k]
        for f in ("excluded_nonfinite", "excluded_step_order", "excluded_domain", "excluded_jump")
    )
    print(f"\n제외 레코드 총계(전 창 합, 창 간 중복 집계 포함): {excl_total}")

    payload = {
        "analysis": "A-9 Table C.1 recompute (locked confirmation)",
        "spec": {"metric": "unsmoothed max |v_y|*/u_g × cfg.Re", "windows": {"std": 15.5, "t60": [15.5, 51.5], "t80": 71.5},
                 "jump_limit_dystar": JUMP_LIMIT_YSTAR, "reference_Re": ref_re},
        "results": results,
        "all_locked_confirmations_match": not confirm_fail,
        "mismatches": [{"config": c, "target": t, "recomputed": r} for c, t, r in confirm_fail],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {OUT_JSON}")
    print("스크립트 SHA256:", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return 0 if not confirm_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
