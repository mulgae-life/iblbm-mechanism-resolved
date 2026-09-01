"""2입자 침강 실험 러너."""

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
from scenarios.sedimentation import make_two_particle_config_reference_benchmark

OUTPUT_ROOT = FINAL_DATA_ROOT / "two_particle_sedimentation"


def _two_particle_baseline_case() -> dict:
    return {
        "case_id": "TWO_PARTICLE_BASE",
        "sets": ("baseline",),
        "kind": "two_particle",
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "euler_explicit",
        "settling_inertia_model": "explicit_history",
        "incompressible_lbgk": True,
        "output_dir": OUTPUT_ROOT / "baseline",
        "vram_gb": 0.0,
    }


def _two_particle_reference_ablation_case(settling_inertia_model: str) -> dict:
    """reference-faithful lane × internal-mass ablation.

    baseline (`_two_particle_baseline_case`)은 `settling_inertia_model="explicit_history"`
    보유하므로, 여기서는 `none` 한 케이스만 추가해 동일 조건 쌍을 완성한다.
    동일 setup: `incompressible_lbgk=True`, `euler_explicit`, BGK, P4.
    """
    return {
        "case_id": f"TWO_PARTICLE_REFERENCE_{settling_inertia_model.upper()}",
        "sets": ("reference_ablation",),
        "kind": "two_particle",
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "euler_explicit",
        "settling_inertia_model": settling_inertia_model,
        "incompressible_lbgk": True,
        "output_dir": OUTPUT_ROOT / "reference_ablation"
                      / f"reference_{settling_inertia_model.lower()}",
        "vram_gb": 0.0,
    }


def _two_particle_matrix_case(
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
        "kind": "two_particle",
        "ibm_method": ibm_method,
        "collision_model": collision_model,
        "delta_type": "peskin4pt",
        "time_integrator": time_integrator,
        "settling_inertia_model": settling_inertia_model,
        "incompressible_lbgk": False,
        "output_dir": OUTPUT_ROOT / "method_matrix" / output_tag,
        "vram_gb": 0.0,
    }


_MATRIX_IBM_METHODS = ("DF", "MDF", "DFC")
_MATRIX_COLLISION_MODELS = ("BGK", "TRT", "CM_MRT")
_MATRIX_TIME_INTEGRATORS = ("verlet", "euler_explicit")
_MATRIX_INERTIA_MODELS = ("none", "explicit_history", "feng_b2", "full_volume")


def _two_particle_extended_60D_case(settling_inertia_model: str, ibm_method: str = "DF") -> dict:
    """DKT 60D extended-domain check (도메인 민감도 분석).

    Majumder default 50D streamwise → 60D 확장 (xmax=5.0 → 6.0).
    locked Peskin 4-point / Velocity-Verlet method-matrix lane을 follow
    하여 baseline 과 직접 비교 가능하도록 setup.
    ibm_method="DF"는 기존 case_id·경로를 그대로 유지하고(비회귀),
    MDF/DFC는 N-1 (60D cross-IBM) 케이스.
    """
    if ibm_method == "DF":
        case_id = f"TWO_PARTICLE_EXT60D_{settling_inertia_model.upper()}"
    else:
        case_id = f"TWO_PARTICLE_EXT60D_{ibm_method}_{settling_inertia_model.upper()}"
    return {
        "case_id": case_id,
        "sets": ("extended_60D",),
        "kind": "two_particle",
        "ibm_method": ibm_method,
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": settling_inertia_model,
        "incompressible_lbgk": False,
        "streamwise_extent_factor": 1.2,
        "output_dir": OUTPUT_ROOT / "extended_60D"
                      / f"{ibm_method.lower()}_bgk_verlet_{settling_inertia_model.lower()}_60D",
        "vram_gb": 0.0,
    }


def _two_particle_extended_80D_case() -> dict:
    """N-8: native 80D right-censoring 진단 1런 (t=0부터, checkpoint 연장 아님).

    단일-arm 진단 전용 — 60D eh와의 unmatched paired contrast 구성 금지.
    마지막 confirmatory extension (80D 단일 진단 런으로 사전 고정).
    """
    return {
        "case_id": "TWO_PARTICLE_EXT80D_NONE",
        "sets": ("extended_80D",),
        "kind": "two_particle",
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": "none",
        "incompressible_lbgk": False,
        "streamwise_extent_factor": 1.6,
        "output_dir": OUTPUT_ROOT / "extended_80D" / "df_bgk_verlet_none_80D",
        "vram_gb": 0.0,
    }


def _two_particle_grid_case(nn_target: int, settling_inertia_model: str) -> dict:
    """N-4: wake-interaction 격자 스윕 (확산 스케일링, τ=0.59 고정).

    κ = (nn_target-1)/800. builder가 Δt∝Δx², g_lat, max_steps, check_interval
    (공통 t* cadence)을 함께 연동한다.
    """
    kappa = (nn_target - 1) / 800.0
    return {
        "case_id": f"TWO_PARTICLE_GRID_NY{nn_target}_{settling_inertia_model.upper()}",
        "sets": ("grid_sensitivity",),
        "kind": "two_particle",
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": settling_inertia_model,
        "incompressible_lbgk": False,
        "grid_refinement_factor": kappa,
        "output_dir": OUTPUT_ROOT / "grid_sensitivity"
                      / f"df_bgk_verlet_{settling_inertia_model.lower()}_ny{nn_target}",
        "vram_gb": 0.0,
    }


def _two_particle_isolated_light_case(settling_inertia_model: str) -> dict:
    """N-7: matched isolated-light configuration.

    동일 도메인·격자·τ·적분기에서 heavy 제거 — 1원소 particles_config로
    다입자 런타임 유지 (단일입자 런타임 전환 금지).
    """
    return {
        "case_id": f"TWO_PARTICLE_ISOLATED_LIGHT_{settling_inertia_model.upper()}",
        "sets": ("isolated_light",),
        "kind": "two_particle",
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": settling_inertia_model,
        "incompressible_lbgk": False,
        "particle_selection": "light_only",
        "output_dir": OUTPUT_ROOT / "isolated_light"
                      / f"df_bgk_verlet_light_{settling_inertia_model.lower()}",
        "vram_gb": 0.0,
    }


def _two_particle_gate_801_both_case() -> dict:
    """G-2 비회귀 게이트: builder 확장 후 801/both 기준 재현 런.

    method_matrix df_bgk_verlet_explicit_history와 동일 builder 인자 —
    기존 peak Re_f 재현으로 particle_selection·grid 확장의 기본 경로 무영향 확인.
    """
    return {
        "case_id": "TWO_PARTICLE_GATE_801_BOTH_EH",
        "sets": ("gates",),
        "kind": "two_particle",
        "ibm_method": "DF",
        "collision_model": "BGK",
        "delta_type": "peskin4pt",
        "time_integrator": "verlet",
        "settling_inertia_model": "explicit_history",
        "incompressible_lbgk": False,
        "output_dir": OUTPUT_ROOT / "gates" / "g2_801_both_eh",
        "vram_gb": 0.0,
    }


TWO_PARTICLE_CASES = [
    _two_particle_baseline_case(),
    *[
        _two_particle_matrix_case(
            (
                f"TWO_PARTICLE_{ibm_method}_{collision_model}_"
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
    _two_particle_reference_ablation_case("none"),
    _two_particle_extended_60D_case("explicit_history"),
    _two_particle_extended_60D_case("none"),
    # --- N-계열 런 ---
    _two_particle_extended_60D_case("explicit_history", "MDF"),
    _two_particle_extended_60D_case("none", "MDF"),
    _two_particle_extended_60D_case("explicit_history", "DFC"),
    _two_particle_extended_60D_case("none", "DFC"),
    _two_particle_grid_case(961, "explicit_history"),
    _two_particle_grid_case(961, "none"),
    _two_particle_grid_case(1281, "explicit_history"),
    _two_particle_grid_case(1281, "none"),
    _two_particle_isolated_light_case("explicit_history"),
    _two_particle_isolated_light_case("none"),
    _two_particle_extended_80D_case(),
    _two_particle_gate_801_both_case(),
]


def _config_signature(case: dict, cfg: SimConfig) -> str:
    return make_signature({
        "case_id": case["case_id"],
        "NN": cfg.NN,
        "Re": getattr(cfg, "Re", None),
        "bc_type": getattr(cfg, "bc_type", None),
        "ibm_method": getattr(cfg, "ibm_method", None),
        "collision_model": getattr(cfg, "collision_model", None),
        "delta_type": getattr(cfg, "delta_type", None),
        "time_integrator": getattr(cfg, "time_integrator", None),
        "settling_inertia_model": getattr(cfg, "settling_inertia_model", None),
        "rotation_coupling": getattr(cfg, "rotation_coupling", None),
        "particles_config": getattr(cfg, "particles_config", None),
        "sedimentation_reference_basis": getattr(cfg, "sedimentation_reference_basis", None),
        "sedimentation_stop_offset_d": getattr(cfg, "sedimentation_stop_offset_d", None),
        "marker_spacing_factor": getattr(cfg, "marker_spacing_factor", None),
        "incompressible_lbgk": getattr(cfg, "incompressible_lbgk", None),
        "max_steps": getattr(cfg, "max_steps", None),
        "check_interval": getattr(cfg, "check_interval", None),
        "xmax": getattr(cfg, "xmax", None),
    })


def _build_config(case: dict) -> SimConfig:
    return make_two_particle_config_reference_benchmark(
        ibm_method=case["ibm_method"],
        collision_model=case["collision_model"],
        delta_type=case["delta_type"],
        rotation_coupling="semi_implicit",
        enable_rotation=True,
        mdf_iterations=20 if case["ibm_method"] == "MDF" else 1,
        time_integrator=case["time_integrator"],
        marker_spacing_factor=0.83,
        incompressible_lbgk=case["incompressible_lbgk"],
        settling_inertia_model=case["settling_inertia_model"],
        sedimentation_stop_offset_d=2.0,
        streamwise_extent_factor=case.get("streamwise_extent_factor", 1.0),
        particle_selection=case.get("particle_selection", "both"),
        grid_refinement_factor=case.get("grid_refinement_factor", 1.0),
        verbose=False,
    )


for _case in TWO_PARTICLE_CASES:
    _case["vram_gb"] = estimate_runtime_vram_gb(_build_config(_case))


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
        "run_path": "two_particle_sedimentation_experiments",
        "config_signature": _config_signature(case, cfg),
        "final_step": result.get("final_step"),
        "termination_reason": result.get("termination_reason"),
        "config": {
            **data.get("config", {}),
            "NN": cfg.NN,
            "delta_type": cfg.delta_type,
            "ibm_method": cfg.ibm_method,
            "collision_model": cfg.collision_model,
            "time_integrator": cfg.time_integrator,
            "settling_inertia_model": cfg.settling_inertia_model,
            "rotation_coupling": cfg.rotation_coupling,
            "sedimentation_reference_basis": cfg.sedimentation_reference_basis,
            "sedimentation_stop_offset_d": cfg.sedimentation_stop_offset_d,
            "incompressible_lbgk": cfg.incompressible_lbgk,
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
            snapshot_every=50,
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
    cases = list(TWO_PARTICLE_CASES)
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
    print("sets | case_id | ibm | collision | inertia | integrator | output_dir")
    print("---|---|---|---|---|---|---")
    for case in cases:
        print(
            f"{','.join(case['sets'])} | {case['case_id']} | {case['ibm_method']} | "
            f"{case['collision_model']} | {case['settling_inertia_model']} | {case['time_integrator']} | "
            f"{case['output_dir']}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Two-particle sedimentation experiments")
    parser.add_argument("--set", nargs="*", default=None, help="실행할 set 목록: baseline, matrix, reference_ablation, extended_60D, extended_80D, grid_sensitivity, isolated_light, gates")
    parser.add_argument("--run", nargs="*", default=None, help="실행할 case_id 목록")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--cpu", action="store_true")
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
            max_vram_gb=7.0,
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
