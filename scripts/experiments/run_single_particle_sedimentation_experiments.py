"""단일 입자 침강 실험 러너."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

if __package__:
    from ._runner_common import (
        FINAL_DATA_ROOT,
        ROOT,
        estimate_runtime_vram_gb,
        is_completed,
        make_signature,
        run_parallel,
    )
else:
    from _runner_common import (
        FINAL_DATA_ROOT,
        ROOT,
        estimate_runtime_vram_gb,
        is_completed,
        make_signature,
        run_parallel,
    )

from iblbm.config import SimConfig
from iblbm.solver import run
from monitor.sedimentation import create_sedimentation_callback
from scenarios.sedimentation import make_sedimentation_config

OUTPUT_ROOT = FINAL_DATA_ROOT / "single_particle_sedimentation"


def _single_baseline_case(rho_ratio: float) -> dict:
    rho_tag = f"rho{int(round(rho_ratio * 100)):03d}"
    return {
        "case_id": f"SINGLE_{rho_tag.upper()}_BASE",
        "sets": ("baseline",),
        "kind": "single_particle",
        "rho_ratio": rho_ratio,
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "euler_explicit",
        "settling_inertia_model": "explicit_history",
        "output_dir": OUTPUT_ROOT / "baseline" / rho_tag,
        "vram_gb": 0.0,
    }


def _single_matrix_case(
    case_id: str,
    settling_inertia_model: str,
    collision_model: str,
    ibm_method: str,
    time_integrator: str,
) -> dict:
    integrator_tag = "euler" if time_integrator == "euler_explicit" else "verlet"
    output_tag = f"{ibm_method.lower()}_{collision_model.lower()}_{integrator_tag}_{settling_inertia_model.lower()}"
    return {
        "case_id": case_id,
        "sets": ("matrix",),
        "kind": "single_particle",
        "rho_ratio": 1.50,
        "ibm_method": ibm_method,
        "collision_model": collision_model,
        "delta_type": "peskin4pt",
        "time_integrator": time_integrator,
        "settling_inertia_model": settling_inertia_model,
        "output_dir": OUTPUT_ROOT / "method_matrix" / "rho150" / output_tag,
        "vram_gb": 0.0,
    }


_MATRIX_IBM_METHODS = ("DF", "MDF", "DFC")
_MATRIX_COLLISION_MODELS = ("BGK", "TRT", "CM_MRT")
_MATRIX_TIME_INTEGRATORS = ("verlet", "euler_explicit")
_MATRIX_INERTIA_MODELS = ("none", "explicit_history", "feng_b2", "full_volume")


def _single_xkernel_case(rho_ratio: float, ibm_method: str, delta_type: str) -> dict:
    """§3.3 Cross-Kernel × Method Verification — verlet+explicit_history+BGK 고정.

    논문 Table VI (ρ_r × method × kernel = 18 lane) 정합화. ρ_r=1.5/DF/peskin4pt
    는 §3.2 `df_bgk_verlet_explicit_history`와 동등 → §3.3 generator에서 제외 (재실행 회피).
    매트릭스 표에서는 §3.2 cell 인용으로 18 lane 표기.
    """
    rho_tag = f"rho{int(round(rho_ratio * 100)):03d}"
    kernel_tag = "hat" if delta_type == "hat" else "p4"
    output_tag = f"{ibm_method.lower()}_bgk_verlet_explicit_history_{kernel_tag}"
    return {
        "case_id": (
            f"SINGLE_{rho_tag.upper()}_{ibm_method}_BGK_VERLET_EXPLICIT_HISTORY_"
            f"{kernel_tag.upper()}"
        ),
        "sets": ("matrix_xkernel",),
        "kind": "single_particle",
        "rho_ratio": rho_ratio,
        "ibm_method": ibm_method,
        "collision_model": "BGK",
        "delta_type": delta_type,
        "time_integrator": "verlet",
        "settling_inertia_model": "explicit_history",
        "output_dir": OUTPUT_ROOT / "method_matrix" / rho_tag / output_tag,
        "vram_gb": 0.0,
        "snapshot_every": 0,  # §3.3은 vy* table만 필요 (디스크 절약)
    }


_XKERNEL_RHO_RATIOS = (1.01, 1.10, 1.50)
_XKERNEL_IBM_METHODS = ("DF", "MDF", "DFC")
_XKERNEL_DELTA_TYPES = ("hat", "peskin4pt")


def _single_rho125_ablation_case(settling_inertia_model: str) -> dict:
    """ρ_r=1.25 density-matched lighter analog control.

    본문 §5.1 / §7.2 / Conclusion 4가 예고한 대조군이다. DF + BGK + Peskin 4-pt +
    Velocity-Verlet 잠금, settling_inertia_model만 ablation.
    """
    output_tag = f"df_bgk_verlet_{settling_inertia_model.lower()}"
    return {
        "case_id": (
            f"SINGLE_RHO125_DF_BGK_VERLET_{settling_inertia_model.upper()}"
        ),
        "sets": ("rho125_ablation",),
        "kind": "single_particle",
        "rho_ratio": 1.25,
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": settling_inertia_model,
        "output_dir": OUTPUT_ROOT / "method_matrix" / "rho125" / output_tag,
        "vram_gb": 0.0,
    }


_RHO125_ABLATION_INERTIA = ("explicit_history", "none")


def _single_density_ablation_case(rho_ratio: float, settling_inertia_model: str) -> dict:
    """ρ=1.01 / ρ=1.10 density-matched lighter additional ablation control.

    §5.1 density-matched lighter band 보강: 기존 ρ=1.25 에 ρ=1.10 과 ρ=1.01 을 추가한다.
    DF / BGK / Peskin 4-pt / Velocity-Verlet 잠금, settling_inertia_model만 ablation.
    """
    rho_tag = f"rho{int(round(rho_ratio * 100)):03d}"  # rho101 / rho110
    output_tag = f"df_bgk_verlet_{settling_inertia_model.lower()}"
    return {
        "case_id": (
            f"SINGLE_{rho_tag.upper()}_DF_BGK_VERLET_{settling_inertia_model.upper()}"
        ),
        "sets": (f"{rho_tag}_ablation",),
        "kind": "single_particle",
        "rho_ratio": rho_ratio,
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": settling_inertia_model,
        "output_dir": OUTPUT_ROOT / "method_matrix" / rho_tag / output_tag,
        "vram_gb": 0.0,
    }


_DENSITY_ABLATION_RHOS = (1.01, 1.10)


def _single_marker_offset_case(retraction_dx: float) -> dict:
    """N-3: marker-offset (D_eff) 시험 — 안쪽 전용 마커 후퇴.

    기존 `retraction_dx` 계약 사용 (코드 변경 0, builder 파라미터 전달만).
    ret00은 G-1 비회귀 게이트 — method_matrix df_bgk_verlet_explicit_history 재현.
    """
    tag = f"ret{int(round(retraction_dx * 10)):02d}"
    return {
        "case_id": f"SINGLE_RHO150_MARKER_OFFSET_{tag.upper()}",
        "sets": ("marker_offset",),
        "kind": "single_particle",
        "rho_ratio": 1.50,
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": "explicit_history",
        "retraction_dx": retraction_dx,
        "output_dir": OUTPUT_ROOT / "marker_offset" / f"rho150_df_bgk_verlet_eh_{tag}",
        "vram_gb": 0.0,
    }


def _single_rho125_nu001_case(settling_inertia_model: str) -> dict:
    """N-6: high-Ga / Re-targeted no-wake control.

    ρ_r=1.25 + ν=0.01 cm²/s (nu_phys_override — 기존 1.25 컨트롤의 ν=0.1과 별도 lane).
    판정 대역 [215,263](예측 [180,300])은 실행 전에 고정했다.
    """
    output_tag = f"df_bgk_verlet_{settling_inertia_model.lower()}"
    return {
        "case_id": f"SINGLE_RHO125_NU001_DF_BGK_VERLET_{settling_inertia_model.upper()}",
        "sets": ("rho125_nu001",),
        "kind": "single_particle",
        "rho_ratio": 1.25,
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": settling_inertia_model,
        "nu_phys_override": 0.01,
        "output_dir": OUTPUT_ROOT / "method_matrix" / "rho125_nu001" / output_tag,
        "vram_gb": 0.0,
    }


def _xkernel_cases() -> list[dict]:
    """§3.3 17 lane = 3 ρ_r × 3 method × 2 kernel − 1 (ρ_r=1.5/DF/peskin4pt → §3.2 인용)."""
    cases = []
    for rho_ratio in _XKERNEL_RHO_RATIOS:
        for ibm_method in _XKERNEL_IBM_METHODS:
            for delta_type in _XKERNEL_DELTA_TYPES:
                if (
                    abs(rho_ratio - 1.50) < 1e-9
                    and ibm_method == "DF"
                    and delta_type == "peskin4pt"
                ):
                    continue
                cases.append(_single_xkernel_case(rho_ratio, ibm_method, delta_type))
    return cases


SINGLE_PARTICLE_CASES = [
    _single_baseline_case(1.01),
    _single_baseline_case(1.10),
    _single_baseline_case(1.50),
    *[
        _single_matrix_case(
            (
                f"SINGLE_RHO150_{ibm_method}_{collision_model}_"
                f"{time_integrator.upper()}_{settling_inertia_model.upper()}"
            ),
            settling_inertia_model,
            collision_model,
            ibm_method,
            time_integrator,
        )
        for ibm_method in _MATRIX_IBM_METHODS
        for collision_model in _MATRIX_COLLISION_MODELS
        for time_integrator in _MATRIX_TIME_INTEGRATORS
        for settling_inertia_model in _MATRIX_INERTIA_MODELS
    ],
    *_xkernel_cases(),
    *[_single_rho125_ablation_case(im) for im in _RHO125_ABLATION_INERTIA],
    *[_single_density_ablation_case(rho, im)
      for rho in _DENSITY_ABLATION_RHOS
      for im in ("none",)],
    # --- N-3 (+ G-1 게이트) 및 N-6 런 ---
    *[_single_marker_offset_case(r) for r in (0.0, 0.3, 0.5, 1.0)],
    *[_single_rho125_nu001_case(im) for im in ("explicit_history", "none")],
]


def _config_signature(case: dict, cfg: SimConfig) -> str:
    return make_signature({
        "case_id": case["case_id"],
        "NN": cfg.NN,
        "Re": getattr(cfg, "Re", None),
        "rho_ratio": getattr(cfg, "rho_ratio", None),
        "bc_type": getattr(cfg, "bc_type", None),
        "ibm_method": getattr(cfg, "ibm_method", None),
        "collision_model": getattr(cfg, "collision_model", None),
        "delta_type": getattr(cfg, "delta_type", None),
        "time_integrator": getattr(cfg, "time_integrator", None),
        "settling_inertia_model": getattr(cfg, "settling_inertia_model", None),
        "sedimentation_reference_basis": getattr(cfg, "sedimentation_reference_basis", None),
        "sedimentation_stop_at_contact": getattr(cfg, "sedimentation_stop_at_contact", None),
        "marker_spacing_factor": getattr(cfg, "marker_spacing_factor", None),
        "max_steps": getattr(cfg, "max_steps", None),
        "check_interval": getattr(cfg, "check_interval", None),
    })


def _build_config(case: dict) -> SimConfig:
    reference_basis = "particle_basis" if abs(case["rho_ratio"] - 1.50) < 1e-9 else "standard"
    return make_sedimentation_config(
        rho_ratio=case["rho_ratio"],
        ibm_method=case["ibm_method"],
        delta_type=case["delta_type"],
        collision_model=case["collision_model"],
        NN=1281,
        verbose=False,
        enable_rotation=True,
        time_integrator=case["time_integrator"],
        incompressible_lbgk=False,
        settling_inertia_model=case["settling_inertia_model"],
        sedimentation_stop_at_contact=True,
        sedimentation_reference_basis=reference_basis,
        diagnostics_interval=0,
        mdf_iterations=20 if case["ibm_method"] == "MDF" else 1,
        retraction_dx=case.get("retraction_dx", 0.0),
        nu_phys_override=case.get("nu_phys_override"),
    )


for _case in SINGLE_PARTICLE_CASES:
    _case["vram_gb"] = estimate_runtime_vram_gb(_build_config(_case))
    # 실측 기반 override — RTX 2070 SUPER 8 GB에서 §3.3 matrix_xkernel
    # nvidia-smi 측정: DF case 2,214 MiB ≈ 2.21 GB.
    # MDF/DFC는 iter buffer로 ~10% 증가 추정 → 2.5 GB 보수.
    # max_vram_gb=7.5 + DF 2.3 / MDF·DFC 2.5 → 어떤 조합이든 3 case 동시 fit.
    if "matrix_xkernel" in _case.get("sets", ()):
        _case["vram_gb"] = 2.3 if _case["ibm_method"] == "DF" else 2.5
    # rho125_ablation: rho150 single과 동일 setup → 실측 ~2.2~2.4 GB.
    # estimate 2.8 GB는 보수치. 실측 기반 override로 2 case 동시 launch fit.
    if "rho125_ablation" in _case.get("sets", ()):
        _case["vram_gb"] = 2.4
    # rho101_ablation / rho110_ablation: rho150과 동일 setup → 실측 ~2.4 GB.
    for tag in ("rho101_ablation", "rho110_ablation"):
        if tag in _case.get("sets", ()):
            _case["vram_gb"] = 2.4


def _write_error_status(case: dict, cfg: SimConfig | None, exc: Exception) -> None:
    payload = {
        "completed": False,
        "case_id": case["case_id"],
        "kind": case["kind"],
        "error": str(exc),
    }
    if cfg is not None:
        payload["config_signature"] = _config_signature(case, cfg)
    with open(case["output_dir"] / "status.json", "w") as f:
        json.dump(payload, f, indent=2)


def _augment_status(case: dict, cfg: SimConfig, result: dict) -> None:
    status_path = case["output_dir"] / "status.json"
    if not status_path.exists():
        return
    with open(status_path) as f:
        data = json.load(f)
    data.update({
        "case_id": case["case_id"],
        "kind": case["kind"],
        "run_path": "single_particle_sedimentation_experiments",
        "config_signature": _config_signature(case, cfg),
        "final_step": result.get("final_step"),
        "termination_reason": result.get("termination_reason"),
        "config": {
            **data.get("config", {}),
            "NN": cfg.NN,
            "rho_ratio": cfg.rho_ratio,
            "delta_type": cfg.delta_type,
            "ibm_method": cfg.ibm_method,
            "collision_model": cfg.collision_model,
            "time_integrator": cfg.time_integrator,
            "settling_inertia_model": cfg.settling_inertia_model,
            "sedimentation_reference_basis": cfg.sedimentation_reference_basis,
        },
    })
    with open(status_path, "w") as f:
        json.dump(data, f, indent=2)


def _run_case(case: dict) -> bool:
    case["output_dir"].mkdir(parents=True, exist_ok=True)
    cfg = None
    try:
        cfg = _build_config(case)
        callback = create_sedimentation_callback(
            str(case["output_dir"]),
            cfg,
            plot_every=10,
            save_frames=False,
            snapshot_every=case.get("snapshot_every", 50),
        )
        result = run(cfg, verbose=True, callback=callback, save_dir=str(case["output_dir"]))
        callback.finalize(
            converged=result.get("converged", False),
            final_state=result["state"],
            final_step=result.get("final_step"),
            termination_reason=result.get("termination_reason"),
        )
        _augment_status(case, cfg, result)
        return True
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _write_error_status(case, cfg, exc)
        return False


def _selected_cases(run_ids: list[str] | None, set_ids: list[str] | None) -> list[dict]:
    cases = list(SINGLE_PARTICLE_CASES)
    if set_ids:
        wanted_sets = set(set_ids)
        cases = [case for case in cases if wanted_sets.intersection(case["sets"])]
    if run_ids:
        wanted_runs = {rid.upper() for rid in run_ids}
        cases = [case for case in cases if case["case_id"] in wanted_runs]
        if len(cases) != len(wanted_runs):
            found = {case["case_id"] for case in cases}
            missing = sorted(wanted_runs - found)
            raise ValueError(f"Unknown case_id(s): {missing}")
    return cases


def _print_case_table(cases: list[dict]) -> None:
    print("sets | case_id | rho_ratio | ibm | collision | delta | inertia | integrator | output_dir")
    print("---|---|---|---|---|---|---|---|---")
    for case in cases:
        print(
            f"{','.join(case['sets'])} | {case['case_id']} | {case['rho_ratio']} | "
            f"{case['ibm_method']} | {case['collision_model']} | {case['delta_type']} | "
            f"{case['settling_inertia_model']} | {case['time_integrator']} | {case['output_dir']}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Single-particle sedimentation experiments")
    parser.add_argument("--set", nargs="*", default=None, help="실행할 set 목록: baseline, matrix, matrix_xkernel, rho101_ablation, rho110_ablation, rho125_ablation, rho125_nu001, marker_offset")
    parser.add_argument("--run", nargs="*", default=None, help="실행할 case_id 목록")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-vram-gb", type=float, default=7.0,
                        help="병렬 실행 시 동시 VRAM 합 한계. 8 GB GPU 기준 디폴트 7.0.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cpu:
        os.environ["IBLBM_GPU"] = "0"

    cases = _selected_cases(args.run, args.set)
    if args.skip_completed:
        cases = [
            case for case in cases
            if not is_completed(case["output_dir"], _config_signature(case, _build_config(case)))
        ]

    if args.list or args.dry_run:
        _print_case_table(cases)
        return 0

    if not cases:
        print("No cases selected.")
        return 0

    if args.parallel and len(cases) > 1:
        ok = run_parallel(
            script_path=Path(__file__),
            cases=cases,
            cpu=args.cpu,
            max_vram_gb=args.max_vram_gb,
        )
        return 0 if ok else 1

    ok = True
    for case in cases:
        print(f"\n{'=' * 72}")
        print(f"[{case['case_id']}] {case['output_dir']}")
        print(f"{'=' * 72}")
        ok = _run_case(case) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
