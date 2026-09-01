"""침강 전용 모니터링 콜백.

solver.run()의 callback으로 호출되어:
  1. state.particle_pos/vel에서 무차원화 계산
  2. status.json에 침강 전용 메트릭 저장
  3. 침강 전용 플롯 생성 (궤적, 속도 시계열, 유동장+입자)
  4. 프레임 저장 (영상 생성용)
  5. velocity field .npz 스냅샷 저장 (streamline/vorticity 후처리용)

기존 server.py의 create_monitoring_callback()과 독립.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from .plots import (
    plot_sedimentation_particle,
    plot_sedimentation_summary,
    plot_sedimentation_trajectory,
    plot_sedimentation_velocity,
)


def _format_elapsed(seconds: float) -> str:
    """초를 'Xh Ym Zs' 형태로 포맷."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _save_status(output_dir, data):
    """status.json 저장 (atomic write)."""
    path = os.path.join(output_dir, "status.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _new_mdf_iter_accumulator():
    """MDF iteration 누적 통계 컨테이너.

    매 step state._mdf_iter_stats 값을 합산해 평균/최대/최소/히스토그램/발산·수렴 카운트 추적.
    histogram bin index = iter_count (0~20). MDF max=20 가정 (cfg.mdf_iterations 디폴트).
    """
    return {
        "n_steps": 0,
        "iter_sum": 0,
        "iter_max": 0,
        "iter_min": -1,  # -1 sentinel: 첫 record 만나면 채움
        "diverge_count": 0,
        "converge_count": 0,
        "histogram": [0] * 21,
        "residual_sum": 0.0,
        "residual_max": 0.0,
    }


def _accumulate_mdf_iter(acc, stats):
    """state._mdf_iter_stats 한 record를 누적기에 더함."""
    if stats is None:
        return
    ic = int(stats.get("iter_count", 0))
    res = float(stats.get("last_residual", 0.0))
    acc["n_steps"] += 1
    acc["iter_sum"] += ic
    if ic > acc["iter_max"]:
        acc["iter_max"] = ic
    if acc["iter_min"] < 0 or ic < acc["iter_min"]:
        acc["iter_min"] = ic
    if 0 <= ic < len(acc["histogram"]):
        acc["histogram"][ic] += 1
    if stats.get("diverged"):
        acc["diverge_count"] += 1
    if stats.get("converged"):
        acc["converge_count"] += 1
    if res != float("inf"):
        acc["residual_sum"] += res
        if res > acc["residual_max"]:
            acc["residual_max"] = res


def _summarize_mdf_iter(acc):
    """status.json용 요약 dict. n_steps=0이면 None."""
    n = acc["n_steps"]
    if n == 0:
        return None
    return {
        "n_steps": n,
        "iter_avg": round(acc["iter_sum"] / n, 3),
        "iter_max": int(acc["iter_max"]),
        "iter_min": int(acc["iter_min"]) if acc["iter_min"] >= 0 else 0,
        "diverge_count": int(acc["diverge_count"]),
        "converge_count": int(acc["converge_count"]),
        "residual_avg": round(acc["residual_sum"] / n, 6),
        "residual_max": round(acc["residual_max"], 6),
        "histogram": list(acc["histogram"]),
    }


def _asnumpy(a):
    """CuPy 배열이면 NumPy로 변환."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _compute_re_metrics(speed, lattice_D, nu_lat, rho_ratio, reference_basis):
    if nu_lat <= 0.0:
        re_standard = 0.0
    else:
        re_standard = float(speed * lattice_D / nu_lat)
    re_particle_basis = float(re_standard * rho_ratio)
    re_reference = re_particle_basis if reference_basis == "particle_basis" else re_standard
    return {
        "Re_standard": round(re_standard, 2),
        "Re_particle_basis": round(re_particle_basis, 2),
        "Re_reference": round(re_reference, 2),
    }


def _extract_peak_re_metrics(history, lattice_D, nu_lat, rho_ratios, reference_basis):
    if not history:
        return {
            "Re_standard_peaks": [],
            "Re_particle_basis_peaks": [],
            "Re_reference_peaks": [],
            "peak_steps": [],
        }

    if "particles" in history[0]:
        re_standard_peaks = []
        re_particle_basis_peaks = []
        re_reference_peaks = []
        peak_steps = []
        for pi, rho_ratio in enumerate(rho_ratios):
            peak_speed = 0.0
            peak_step = 0
            for rec in history:
                pdata = rec["particles"][pi]
                speed = float(np.sqrt(pdata["vx"] ** 2 + pdata["vy"] ** 2))
                if speed > peak_speed:
                    peak_speed = speed
                    peak_step = int(rec["step"])
            metrics = _compute_re_metrics(
                peak_speed, lattice_D, nu_lat, rho_ratio, reference_basis,
            )
            re_standard_peaks.append(metrics["Re_standard"])
            re_particle_basis_peaks.append(metrics["Re_particle_basis"])
            re_reference_peaks.append(metrics["Re_reference"])
            peak_steps.append(peak_step)
        return {
            "Re_standard_peaks": re_standard_peaks,
            "Re_particle_basis_peaks": re_particle_basis_peaks,
            "Re_reference_peaks": re_reference_peaks,
            "peak_steps": peak_steps,
        }

    peak_speed = 0.0
    peak_step = 0
    for rec in history:
        speed = float(np.sqrt(rec["vx"] ** 2 + rec["vy"] ** 2))
        if speed > peak_speed:
            peak_speed = speed
            peak_step = int(rec["step"])
    metrics = _compute_re_metrics(
        peak_speed, lattice_D, nu_lat, rho_ratios[0], reference_basis,
    )
    return {
        "Re_standard_peaks": [metrics["Re_standard"]],
        "Re_particle_basis_peaks": [metrics["Re_particle_basis"]],
        "Re_reference_peaks": [metrics["Re_reference"]],
        "peak_steps": [peak_step],
    }


def _save_velocity_snapshot(output_dir, state, step):
    """velocity field .npz 스냅샷 저장 (실린더 실험과 동일 패턴).

    저장 내용: Eux, Euy (float32), dx, dy, step, particle_pos
    streamline/vorticity 후처리에 사용.
    """
    snap_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    ny, nx = state.ny, state.nx
    U = _asnumpy(state.U)
    Eux = U[:, 0].reshape(ny, nx).astype(np.float32)
    Euy = U[:, 1].reshape(ny, nx).astype(np.float32)

    pos = _asnumpy(state.particle_pos) if state.particle_pos is not None else np.zeros(2)

    path = os.path.join(snap_dir, f"velocity_{int(step):07d}.npz")
    np.savez_compressed(
        path,
        Eux=Eux, Euy=Euy,
        dx=state.dx, dy=state.dy,
        step=int(step),
        particle_pos=pos.astype(np.float64),
    )


def create_sedimentation_callback(
    output_dir: str,
    cfg,
    plot_every: int = 10,
    save_frames: bool = False,
    snapshot_every: int = 50,
):
    """침강 전용 모니터링 콜백 생성.

    Args:
        output_dir: 출력 디렉토리
        cfg: SimConfig
        plot_every: 플롯 생성 빈도 (check_interval 단위)
        save_frames: True면 매 콜백마다 유동장+입자 프레임 저장
        snapshot_every: velocity field .npz 저장 빈도 (check_interval 단위).
            0이면 비활성화. 기본 50 = 50×check_interval 스텝마다 저장.

    Returns:
        solver.run()의 callback 파라미터로 전달할 함수
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames") if save_frames else None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    # 무차원화에 필요한 상수 (초기화 시 1회 계산)
    lattice_D = cfg.cylinder_D_ratio * (cfg.NN - 1)
    d_lattice = lattice_D  # 입자 직경 (격자 단위)
    _dx = 1.0 / (cfg.NN - 1)  # 격자 간격 (도메인 좌표)
    D_domain = d_lattice * _dx  # 도메인 좌표 직경
    g_lattice = cfg.gravity
    rho_ratio = cfg.rho_ratio
    x0, y0 = cfg.cylinder_center

    u_g = np.sqrt(abs(rho_ratio - 1.0) * g_lattice * d_lattice)

    # 영상 떨림 방지: 고정 컬러바 범위 (u_g 기반, 격자 단위)
    # state.U는 격자 단위이므로 _dx 변환 불필요
    speed_vmax = float(1.5 * u_g) if u_g > 1e-15 else 0.1

    # 축적 데이터
    history = []
    call_count = [0]
    last_elapsed = [0.0]
    last_state = [None]  # finalize에서 최종 velocity_field.npz 저장용
    mdf_iter_acc = _new_mdf_iter_accumulator()

    is_multiparticle = (hasattr(cfg, 'particles_config')
                        and cfg.particles_config is not None
                        and len(cfg.particles_config) > 1)
    gravity_right = getattr(cfg, 'gravity_direction', 'down') == 'right'

    def callback(step, Cd, Cl, error, state, elapsed, converged):
        """check_interval마다 호출."""
        call_count[0] += 1
        interval_time = elapsed - last_elapsed[0]
        last_elapsed[0] = elapsed
        last_state[0] = state

        if is_multiparticle and state.particles is not None:
            particles_data = []
            for pi, p in enumerate(state.particles):
                pos_p = _asnumpy(p.pos)
                vel_p = _asnumpy(p.vel)
                u_g_p = np.sqrt(abs(p.rho_ratio - 1.0) * g_lattice * d_lattice)
                if u_g_p > 1e-15:
                    t_star_p = float(step * u_g_p / d_lattice)
                    if gravity_right:
                        vx_star_p = float(vel_p[0] / u_g_p)
                        vy_star_p = float(vel_p[1] / u_g_p)
                    else:
                        vx_star_p = float(vel_p[0] / u_g_p)
                        vy_star_p = float(-vel_p[1] / u_g_p)
                else:
                    t_star_p = vx_star_p = vy_star_p = 0.0
                particles_data.append({
                    "id": pi, "rho_ratio": p.rho_ratio,
                    "t_star": t_star_p,
                    "x": float(pos_p[0]), "y": float(pos_p[1]),
                    "vx": float(vel_p[0]), "vy": float(vel_p[1]),
                    "vx_star": vx_star_p, "vy_star": vy_star_p,
                })
            record = {"step": int(step), "particles": particles_data}
            mdf_stats = getattr(state, "_mdf_iter_stats", None)
            if mdf_stats is not None:
                record["mdf_iter"] = int(mdf_stats.get("iter_count", 0))
                record["mdf_residual"] = float(mdf_stats.get("last_residual", 0.0))
                record["mdf_diverged"] = bool(mdf_stats.get("diverged", False))
                _accumulate_mdf_iter(mdf_iter_acc, mdf_stats)
            history.append(record)
            pos = _asnumpy(state.particles[0].pos)
            vel = _asnumpy(state.particles[0].vel)
            vy_star = particles_data[0]["vy_star"]
            vx_star = particles_data[0]["vx_star"]
            if gravity_right:
                y_star = float((pos[0] - x0) / D_domain)
            else:
                y_star = float((y0 - pos[1]) / D_domain)
        else:
            if hasattr(state, 'particles') and state.particles is not None and len(state.particles) > 0:
                pos = _asnumpy(state.particles[0].pos)
                vel = _asnumpy(state.particles[0].vel)
            else:
                pos = state.particle_pos
                vel = state.particle_vel

            if u_g > 1e-15:
                t_star = float(step * u_g / d_lattice)
                if gravity_right:
                    y_star = float((pos[0] - x0) / D_domain)
                    vy_star = float(vel[1] / u_g)
                    vx_star = float(vel[0] / u_g)
                else:
                    y_star = float((y0 - pos[1]) / D_domain)
                    vy_star = float(-vel[1] / u_g)
                    vx_star = float(vel[0] / u_g)
            else:
                t_star = y_star = vy_star = vx_star = 0.0

            record = {
                "t_star": t_star, "y_star": y_star,
                "vy_star": vy_star, "vx_star": vx_star,
                "x": float(pos[0]), "y": float(pos[1]),
                "vx": float(vel[0]), "vy": float(vel[1]),
                "step": int(step),
                "f_hydro_x": float(state.particle_force[0]) if state.particle_force is not None else 0.0,
                "f_hydro_y": float(state.particle_force[1]) if state.particle_force is not None else 0.0,
            }
            if hasattr(state, "particle_angle"):
                record["angle"] = float(state.particle_angle)
            if hasattr(state, "particle_omega"):
                record["omega"] = float(state.particle_omega)
            if hasattr(state, "particle_torque"):
                record["torque"] = float(state.particle_torque)
            if hasattr(state, "inside_residual_mean") and state.inside_residual_mean != 0.0:
                record["inside_residual_mean"] = state.inside_residual_mean
                record["inside_residual_max"] = state.inside_residual_max
            if state.p_int is not None:
                record["p_int_x"] = float(state.p_int[0])
                record["p_int_y"] = float(state.p_int[1])
                record["l_int_z"] = state.l_int
            mdf_stats = getattr(state, "_mdf_iter_stats", None)
            if mdf_stats is not None:
                record["mdf_iter"] = int(mdf_stats.get("iter_count", 0))
                record["mdf_residual"] = float(mdf_stats.get("last_residual", 0.0))
                record["mdf_diverged"] = bool(mdf_stats.get("diverged", False))
                _accumulate_mdf_iter(mdf_iter_acc, mdf_stats)
            history.append(record)
            pos = np.array([float(pos[0]), float(pos[1])])

        # status.json 저장
        status_data = {
            "experiment_type": "sedimentation",
            "completed": False,
            "step": int(step),
            "elapsed_sec": round(elapsed, 1),
            "elapsed_str": _format_elapsed(elapsed),
            "interval_time": round(interval_time, 2),
            "converged": False,
            "vy_star": round(vy_star, 6),
            "y_star": round(y_star, 4),
            "vx_star": round(vx_star, 6),
            "particle_x": round(float(pos[0]), 6),
            "particle_y": round(float(pos[1]), 6),
            "config": {
                "rho_ratio": rho_ratio,
                "ibm_method": cfg.ibm_method,
                "delta_type": getattr(cfg, "delta_type", "peskin4pt"),
                "collision_model": getattr(cfg, "collision_model", "BGK"),
                "settling_inertia_model": getattr(cfg, "settling_inertia_model", "unset"),
                "NN": cfg.NN,
                "max_steps": cfg.max_steps,
                "check_interval": cfg.check_interval,
                "gravity": g_lattice,
                "gravity_direction": getattr(cfg, "gravity_direction", "down"),
                "time_integrator": getattr(cfg, "time_integrator", "verlet"),
                "enable_rotation": getattr(cfg, "enable_rotation", False),
                "rotation_coupling": getattr(cfg, "rotation_coupling", "indirect"),
                "marker_spacing_factor": getattr(cfg, "marker_spacing_factor", 1.0),
                "retraction_dx": getattr(cfg, "retraction_dx", 0.0),
                "mdf_iterations": getattr(cfg, "mdf_iterations", 1),
                "incompressible_lbgk": getattr(cfg, "incompressible_lbgk", False),
                "sedimentation_stop_at_contact": getattr(cfg, "sedimentation_stop_at_contact", False),
                "sedimentation_stop_offset_d": getattr(cfg, "sedimentation_stop_offset_d", 0.0),
                "sedimentation_reference_basis": getattr(cfg, "sedimentation_reference_basis", "standard"),
                "sedimentation_euler_update_scheme": getattr(cfg, "sedimentation_euler_update_scheme", "new_velocity"),
            },
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        mdf_summary = _summarize_mdf_iter(mdf_iter_acc)
        if mdf_summary is not None:
            status_data["mdf_iter_stats"] = mdf_summary
        _save_status(output_dir, status_data)

        # velocity field 스냅샷 저장 (streamline/vorticity 후처리용)
        if snapshot_every > 0 and call_count[0] % snapshot_every == 0:
            _save_velocity_snapshot(output_dir, state, step)

        # 프레임 저장
        frame_path = None
        if frames_dir:
            frame_path = os.path.join(frames_dir, f"sed_{int(step):07d}.png")

        # 플롯 + history JSON 주기적 저장 (crash 대비)
        if call_count[0] == 1 or call_count[0] % plot_every == 0:
            plot_sedimentation_velocity(output_dir, history)
            plot_sedimentation_trajectory(output_dir, history)
            plot_sedimentation_particle(output_dir, state,
                                        frame_path=frame_path,
                                        speed_vmax=speed_vmax,
                                        gravity_direction=getattr(cfg, "gravity_direction", "down"))
            # history JSON 증분 저장 — finalize() 전에도 데이터 보존
            hist_path = os.path.join(output_dir, "sedimentation_history.json")
            tmp_path = hist_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(history, f, indent=2)
            os.replace(tmp_path, hist_path)
        elif frame_path:
            # 플롯 주기가 아니어도 프레임은 항상 저장
            plot_sedimentation_particle(output_dir, state,
                                        frame_path=frame_path,
                                        speed_vmax=speed_vmax,
                                        gravity_direction=getattr(cfg, "gravity_direction", "down"))

    def finalize(
        converged: bool = False,
        final_state=None,
        final_step: int | None = None,
        termination_reason: str | None = None,
    ):
        """run 완료 후 호출. history JSON + 최종 velocity_field.npz + 요약 플롯."""
        # sedimentation_history.json 저장
        hist_path = os.path.join(output_dir, "sedimentation_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        # 최종 velocity_field.npz 저장 (실린더 실험과 동일 패턴)
        state = final_state if final_state is not None else last_state[0]
        if state is not None:
            ny, nx = state.ny, state.nx
            U = _asnumpy(state.U)
            Eux = U[:, 0].reshape(ny, nx)
            Euy = U[:, 1].reshape(ny, nx)
            pos = _asnumpy(state.particle_pos) if state.particle_pos is not None else np.zeros(2)

            vf_path = os.path.join(output_dir, "velocity_field.npz")
            np.savez_compressed(
                vf_path,
                Eux=Eux, Euy=Euy,
                dx=state.dx, dy=state.dy,
                particle_pos=pos,
            )

        # 최종 요약 플롯
        plot_sedimentation_summary(output_dir, history)

        # status.json에 완료 표시
        status_path = os.path.join(output_dir, "status.json")
        if os.path.exists(status_path):
            with open(status_path) as f:
                data = json.load(f)
            data["completed"] = True
            data["converged"] = converged
            if final_step is not None:
                data["step"] = int(final_step)
                data["final_step"] = int(final_step)
            if termination_reason is not None:
                data["termination_reason"] = termination_reason
            if state is not None:
                lattice_D = cfg.cylinder_D_ratio * (cfg.NN - 1)
                nu_lat = (float(state.tau) - 0.5) / 3.0
                if is_multiparticle and state.particles is not None:
                    rho_ratios = [float(p.rho_ratio) for p in state.particles]
                else:
                    rho_ratios = [float(getattr(cfg, "rho_ratio", 1.0))]
                peak_metrics = _extract_peak_re_metrics(
                    history,
                    lattice_D,
                    nu_lat,
                    rho_ratios,
                    getattr(cfg, "sedimentation_reference_basis", "standard"),
                )
                data.update(peak_metrics)
                data["history_len"] = len(history)
                data["re_reference_basis"] = getattr(
                    cfg, "sedimentation_reference_basis", "standard",
                )
                if peak_metrics["Re_standard_peaks"]:
                    data["Re_standard_peak"] = peak_metrics["Re_standard_peaks"][0]
                    data["Re_particle_basis_peak"] = peak_metrics["Re_particle_basis_peaks"][0]
                    data["Re_reference_peak"] = peak_metrics["Re_reference_peaks"][0]
            _save_status(output_dir, data)

    callback.finalize = finalize
    callback.history = history
    return callback
