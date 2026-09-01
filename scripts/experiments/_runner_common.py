from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("IBLBM_GPU", "1")

ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_DATA_ROOT = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def estimate_runtime_vram_gb(cfg) -> float:
    if cfg.nx_formula == "standard":
        nx = int(cfg.NN * cfg.xmax - (cfg.xmax - 1))
    else:
        nx = int(cfg.NN * cfg.xmax)

    dx = 1.0 / (cfg.NN - 1)
    ny = round(cfg.ymax / dx) + 1
    nodenums = nx * ny

    # initialize() 기준 핵심 격자 배열:
    # fstar, feq, ro, ro_initial, U, fib = 24 float64 / node = 192 B / node
    init_bytes = nodenums * 24 * 8

    # sedimentation은 torque/full-volume 좌표 캐시 4개를 추가로 보유한다.
    if cfg.motion_type == "sedimentation":
        init_bytes += nodenums * 4 * 8

    # 라그랑주 배열은 격자 배열에 비해 작지만 추정식에 포함한다.
    lattice_r = cfg.cylinder_D_ratio * (cfg.NN - 1) / 2.0
    marker_count = max(1, int(round(2.0 * math.pi * lattice_r / cfg.marker_spacing_factor)))
    init_bytes += marker_count * 4 * 8
    if cfg.particles_config is not None:
        init_bytes += len(cfg.particles_config) * marker_count * 6 * 8

    multiplier = 2.2
    if cfg.motion_type == "sedimentation":
        multiplier *= 1.2
    if cfg.settling_inertia_model == "full_volume":
        multiplier *= 1.12
    if cfg.particles_config is not None:
        multiplier *= 1.15
    if cfg.ibm_method in {"MDF", "DFC"}:
        multiplier *= 1.05
    if cfg.collision_model == "CM_MRT":
        multiplier *= 1.03

    vram_gb = init_bytes * multiplier / (1024 ** 3)
    return math.ceil(vram_gb * 10.0) / 10.0


def make_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def load_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def is_completed(output_dir: Path, signature: str) -> bool:
    status = load_status(output_dir / "status.json")
    if status is None:
        return False
    return bool(status.get("completed", False)) and status.get("config_signature") == signature


def pick_launch_batch(pending: list[dict], available_vram_gb: float) -> list[int]:
    capacity = int(round(available_vram_gb * 10))
    if capacity <= 0:
        return []

    dp: dict[int, list[int]] = {0: []}
    for idx, case in enumerate(pending):
        weight = int(round(case["vram_gb"] * 10))
        if weight > capacity:
            continue
        updates: dict[int, list[int]] = {}
        for used, chosen in list(dp.items()):
            nxt = used + weight
            if nxt > capacity:
                continue
            if nxt not in dp and nxt not in updates:
                updates[nxt] = chosen + [idx]
        dp.update(updates)

    return dp[max(dp)]


def run_parallel(
    *,
    script_path: Path,
    cases: list[dict],
    cpu: bool,
    max_vram_gb: float,
) -> bool:
    pending = sorted(cases, key=lambda case: case["vram_gb"], reverse=True)
    active: list[dict] = []
    ok = True

    while pending or active:
        launched = False
        used_vram = sum(item["case"]["vram_gb"] for item in active)
        batch_indices = pick_launch_batch(pending, max_vram_gb - used_vram)
        if batch_indices:
            selected = {idx for idx in batch_indices}
            launch_cases = [case for idx, case in enumerate(pending) if idx in selected]
            pending = [case for idx, case in enumerate(pending) if idx not in selected]
        else:
            launch_cases = []

        for case in launch_cases:
            cmd = [sys.executable, "-u", str(script_path), "--run", case["case_id"]]
            if cpu:
                cmd.append("--cpu")
            proc = subprocess.Popen(cmd, env=os.environ.copy(), cwd=str(ROOT))
            active.append({"case": case, "proc": proc})
            launched = True

        next_active = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                next_active.append(item)
                continue
            ok = ok and (ret == 0)
        active = next_active

        if active and not launched:
            time.sleep(2.0)

    return ok

