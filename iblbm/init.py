"""`SimState` 초기화.

담당 범위
  - 격자 생성 (nx/ny, dx/dy, nodenums)
  - 단위 변환 (`phy_l` → lattice, `cylinder_D_ratio` → lattice_r)
  - Re → ν → τ 관계식
  - 라그랑주 마커 배치 (원주 등분할, retraction 반영)
  - 경계 인덱스, 초기 분포 함수 f_eq = f* 생성
  - 충돌 모델별 state 초기화 (`get_collision(...).init_state(cfg, state)`)
  - 침강 시나리오에서 particle 필드 주입 (`init_sedimentation_state`)
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import xp as np

from .boundary import BoundaryIndices, build_boundary_indices
from .config import SimConfig
from .diagnostics import tg_analytical_velocity_field
from .lbm import compute_feq
from .lbm import D2Q9, make_d2q9
from .lbm.collision import get_collision


@dataclass
class SimState:
    """시뮬레이션 상태 (`initialize()` 반환값).

    필드 그룹
      1. 격자:      `nx`, `ny`, `nodenums`, `dx`, `dy`, `dt`, `tau`
      2. 물리:      `r`, `lattice_r`, `in_u`, `dens`
      3. 분포함수:  `fstar`, `feq`
      4. 거시변수:  `ro`, `ro_initial`, `U`
      5. IB 체적력: `fib`
      6. 라그랑주:  `Lx`, `Ly`, `Lb`, `Larea`, `desired_velocity`
      7. 경계/격자: `idx`, `lattice`
      8. 충돌모델 캐시: `tau_minus` (TRT), `S_diag`/`M`/`M_inv` (CM_MRT)
      9. IBM 캐시:  `lambda_cache`, `dfc_force_lagr`
     10. 침강:     `particle_pos`/`_vel`/`_vel_prev`/`_force`/`_mass`, `gravity_lattice`
     11. 진단 캐시: `inside_residual_mean`/`_max`, `p_int`, `l_int`
     12. 다입자:   `particles` (None → 단일입자 경로)
    """

    # 격자
    nx: int
    ny: int
    nodenums: int
    dx: float
    dy: float
    dt: float
    tau: float

    # 물리 (격자 단위)
    r: float               # 실린더 반지름 (도메인 비율)
    lattice_r: float        # 실린더 반지름 (격자 단위, 실수)
    in_u: float             # 유입 속도
    dens: float             # 격자 밀도 (= 1.0)

    # 분포 함수
    fstar: np.ndarray       # (nodenums, 9) — post-streaming
    feq: np.ndarray         # (nodenums, 9) — 평형

    # 거시 변수
    ro: np.ndarray          # (nodenums,) — 밀도
    ro_initial: np.ndarray  # (nodenums,) — 초기 밀도
    U: np.ndarray           # (nodenums, 2) — 속도

    # IB 체적력
    fib: np.ndarray         # (nodenums, 2)

    # 라그랑주 경계
    Lx: np.ndarray          # (Lb,) — x 좌표
    Ly: np.ndarray          # (Lb,) — y 좌표
    Lb: int                 # 경계점 수
    Larea: float            # 각 점의 호 길이 Δs (lattice units)
    desired_velocity: np.ndarray  # (Lb, 2)

    # 경계 인덱스
    idx: BoundaryIndices

    # 격자
    lattice: D2Q9

    # DFC 캐시 (ibm_method="DFC" 전용)
    lambda_cache: np.ndarray | None = None       # λ(k) 캐시 (고정 경계)
    dfc_force_lagr: np.ndarray | None = None     # 최근 DFC 마커별 힘 집계량 (Lb, 2)

    # TRT 전용
    tau_minus: float | None = None        # TRT 반대칭 이완 시간

    # CM_MRT 전용
    S_diag: np.ndarray | None = None      # (9,) 이완률 벡터
    M: np.ndarray | None = None           # (9,9) 변환 행렬
    M_inv: np.ndarray | None = None       # (9,9) 역변환 행렬

    # 침강 전용 (motion_type="sedimentation")
    particle_pos: np.ndarray | None = None     # (2,) 입자 중심 [x, y] (도메인 좌표)
    particle_vel: np.ndarray | None = None     # (2,) 입자 속도 [vx, vy] (격자 단위)
    particle_vel_prev: np.ndarray | None = None  # (2,) 이전 스텝 입자 속도 [vx, vy]
    particle_force: np.ndarray | None = None   # (2,) 현재 스텝 유체력 [Fx, Fy]
    particle_mass: float = 0.0                 # 격자 단위 입자 질량
    gravity_lattice: float = 0.0               # 격자 단위 중력 가속도 (양수)

    # 내부 유체 진단 캐시 (motion_type="sedimentation", diagnostics_interval > 0)
    inside_residual_mean: float = 0.0
    inside_residual_max: float = 0.0
    p_int: np.ndarray | None = None            # (2,) 내부 선형 운동량
    l_int: float = 0.0                          # 내부 각운동량 (z)

    # 다입자 (particles_config 사용 시)
    particles: list | None = None  # list[ParticleState]. None → 단일입자 기존 경로


def initialize(cfg: SimConfig) -> SimState:
    """`SimConfig` → 완성된 `SimState` 생성.

    단계
      1. D2Q9 lattice 생성
      2. nx, dx, ny, dy 산출 (정사각 격자 검증)
      3. 단위 변환 (`phy_l` → lattice, cylinder 반지름)
      4. Re → ν → τ (회전: ν = r²·ω / Re, 그 외: ν = u·D / Re)
      5. 경계 인덱스 / 초기 거시변수 (시나리오별 분기)
      6. 초기 분포 함수 feq → fstar 복제
      7. Lagrangian 마커 원주 등분할 (retraction 적용)
      8. 충돌 모델별 `init_state(cfg, state)`
      9. 침강: `init_sedimentation_state(state, cfg, lattice_r)`
    """
    lattice = make_d2q9()

    # --- 격자 생성 ---
    if cfg.nx_formula == "standard":
        # 정상류/회전: nx = NN*xmax - (xmax-1) → dx=dy 보장
        nx = int(cfg.NN * cfg.xmax - (cfg.xmax - 1))
    else:
        # 진동: nx = NN*xmax → dx ≈ dy (0.08% 차이)
        nx = int(cfg.NN * cfg.xmax)

    dx = 1.0 / (cfg.NN - 1)

    # ny를 ymax 에서 유도 — 정사각 격자 (dx = dy) 보장용
    ny = round(cfg.ymax / dx) + 1
    dy = cfg.ymax / (ny - 1)

    # 정사각 격자 검증
    if abs(dx - dy) / dx > 1e-6:
        raise ValueError(
            f"정사각 격자 불일치: dx={dx:.8f}, dy={dy:.8f}. "
            f"ymax={cfg.ymax}가 dx와 호환 불가"
        )

    nodenums = nx * ny
    dt = 1.0  # 격자 단위 시간 간격

    # --- 단위 변환 ---
    phy_char = cfg.phy_l * cfg.cylinder_D_ratio
    phy_dx = cfg.phy_l / (cfg.NN - 1)
    Cx = phy_dx / 1.0

    r = phy_char / 2.0 / cfg.phy_l
    lattice_D = phy_char / Cx
    lattice_r = lattice_D / 2.0

    # Re → nu → tau
    lattice_nu = cfg.lattice_u * 2.0 * lattice_r / cfg.Re

    tau = 3.0 * lattice_nu + 0.5

    in_u = cfg.inflow_u
    dens = 1.0  # 격자 단위 기준 밀도

    # --- 경계 인덱스 (0-base) ---
    idx = build_boundary_indices(cfg, nx, ny)

    # --- 초기 거시 변수 ---
    U = np.zeros((nodenums, 2))
    if cfg.scenario_type == "taylor_green":
        # TG: 격자 인덱스를 중심 기준 격자 좌표로 변환 (lattice units)
        # 격자 [0, nx-1] → [-L_lat, L_lat], L_lat = (NN-1)/2
        L_lat = 0.5 * (cfg.NN - 1)
        X_c = np.arange(nx, dtype=float) - (nx - 1) / 2.0
        Y_c = np.arange(ny, dtype=float) - (ny - 1) / 2.0
        XX, YY = np.meshgrid(X_c, Y_c)  # (ny, nx) — 격자 좌표
        # t=0 해석해로 초기 속도 설정 (모든 변수 lattice units)
        lattice_D_tg = cfg.cylinder_D_ratio * (cfg.NN - 1)
        nu_tg = cfg.lattice_u * lattice_D_tg / cfg.Re
        ux_ana, uy_ana = tg_analytical_velocity_field(
            XX.ravel(), YY.ravel(), 0.0, cfg.tg_u0, L_lat, nu_tg,
        )
        U[:, 0] = ux_ana
        U[:, 1] = uy_ana
    elif cfg.motion_type == "oscillating":
        pass  # U = 0 (정지 유체)
    elif cfg.motion_type == "sedimentation":
        pass  # U = 0 (정지 유체, oscillating과 동일)
    else:
        U[:, 0] = in_u  # u = inflow_u

    ro = dens * np.ones(nodenums)
    ro_initial = dens * np.ones(nodenums)  # IBM 보간에 전달할 초기 밀도

    # 초기 분포 함수
    feq = compute_feq(ro, U, lattice, incompressible=cfg.incompressible_lbgk)
    fstar = feq.copy()

    # IB 체적력
    fib = np.zeros((nodenums, 2))

    # --- 라그랑주 경계점 ---
    dd = 2.0 * np.pi * lattice_r / cfg.marker_spacing_factor
    step = 1.0 / dd
    Ldx = np.arange(0, 1.0 + step * 0.5, step)  # 매개변수 [0, ~1)
    # 1.0을 넘는 점 제거
    Ldx = Ldx[Ldx <= 1.0 + 1e-12]
    # 마지막 점이 첫 점과 겹치면 제거 (원 닫힘)
    if len(Ldx) > 1 and np.abs(Ldx[-1] - 1.0) < step * 0.5:
        Ldx = Ldx[:-1]

    # 좌표는 도메인 비율 반지름(r) 사용, 격자 단위가 아님
    cx, cy = cfg.cylinder_center

    # 실린더 경계 검사
    if cx - r < 0 or cx + r > cfg.xmax or cy - r < 0 or cy + r > cfg.ymax:
        raise ValueError(
            f"실린더(cx={cx}, cy={cy}, r={r})가 도메인 밖: "
            f"[0,{cfg.xmax}]×[0,{cfg.ymax}]"
        )

    if cfg.retraction_dx < 0.0 or cfg.retraction_dx >= lattice_r:
        raise ValueError(
            f"retraction_dx={cfg.retraction_dx} 범위 초과: "
            f"0.0 <= retraction_dx < lattice_r({lattice_r})"
        )

    theta = 2.0 * np.pi * Ldx
    r_retracted = r - cfg.retraction_dx * dx
    Lx = r_retracted * np.cos(theta) + cx
    Ly = r_retracted * np.sin(theta) + cy
    Lb = len(Lx)
    lattice_r_retracted = lattice_r - cfg.retraction_dx
    Larea = 2.0 * np.pi * lattice_r_retracted / Lb

    desired_velocity = np.zeros((Lb, 2))

    # --- 충돌 모델별 초기화 ---
    state = SimState(
        nx=nx, ny=ny, nodenums=nodenums,
        dx=dx, dy=dy, dt=dt, tau=tau,
        r=r, lattice_r=lattice_r, in_u=in_u, dens=dens,
        fstar=fstar, feq=feq,
        ro=ro, ro_initial=ro_initial, U=U,
        fib=fib,
        Lx=Lx, Ly=Ly, Lb=Lb, Larea=Larea,
        desired_velocity=desired_velocity,
        idx=idx, lattice=lattice,
    )

    get_collision(cfg.collision_model).init_state(cfg, state)

    if cfg.motion_type == "sedimentation":
        from .physics.sedimentation import init_sedimentation_state
        init_sedimentation_state(state, cfg, lattice_r)

    return state
