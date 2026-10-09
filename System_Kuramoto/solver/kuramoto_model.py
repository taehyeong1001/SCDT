"""
Kuramoto Oscillator Model
=========================
N개의 결합된 위상 진동자를 기술하는 쿠라모토 모델.

수식:
    dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i),  i = 1, ..., N

    - θ_i : i번째 진동자의 위상
    - ω_i : i번째 진동자의 고유 진동수 (자연 주파수)
    - K   : 결합 강도 (coupling strength)
    - N   : 진동자 수

동기화 정도 (order parameter):
    r(t) = |1/N * Σ_j exp(i θ_j)|   (0 = 완전 비동기, 1 = 완전 동기)
"""

import numpy as np
from scipy.integrate import solve_ivp


# ──────────────────────────────────────────────
#  ODE 우변 (right-hand side)
# ──────────────────────────────────────────────

def kuramoto_rhs(t, theta, omega, K):
    """쿠라모토 모델의 우변 함수.

    Parameters
    ----------
    t : float
        현재 시각 (solve_ivp 인터페이스 요구사항).
    theta : array_like, shape (N,)
        각 진동자의 현재 위상 [rad].
    omega : array_like, shape (N,)
        각 진동자의 고유 진동수 [rad/s].
    K : float
        결합 강도.

    Returns
    -------
    dtheta_dt : ndarray, shape (N,)
        위상 변화율.
    """
    theta = np.asarray(theta)
    N = len(theta)
    # 결합항: (K/N) * Σ_j sin(θ_j - θ_i)
    # 브로드캐스팅으로 NxN 행렬 계산 후 행 합산
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]   # diff[i,j] = θ_j - θ_i
    coupling = (K / N) * np.sum(np.sin(diff), axis=1)
    return omega + coupling


# ──────────────────────────────────────────────
#  솔버
# ──────────────────────────────────────────────

def solve_kuramoto(
    N: int = 50,
    K: float = 2.0,
    t_span: tuple = (0.0, 50.0),
    dt: float = 0.05,
    omega_mean: float = 0.0,
    omega_std: float = 1.0,
    seed: int = 42,
    theta0: np.ndarray = None,
):
    """쿠라모토 모델을 수치적으로 적분한다.

    Parameters
    ----------
    N : int
        진동자 수.
    K : float
        결합 강도.
    t_span : (float, float)
        시뮬레이션 시간 범위 (t_start, t_end).
    dt : float
        저장 시간 간격.
    omega_mean : float
        고유 진동수 분포 평균 (가우시안).
    omega_std : float
        고유 진동수 분포 표준편차 (가우시안).
    seed : int
        난수 시드.
    theta0 : ndarray, optional
        초기 위상 벡터 (None이면 [-π, π] 균일분포).

    Returns
    -------
    t : ndarray, shape (T,)
        시각 배열.
    theta : ndarray, shape (T, N)
        각 진동자의 위상 궤적.
    omega : ndarray, shape (N,)
        각 진동자에 할당된 고유 진동수.
    r : ndarray, shape (T,)
        시각별 order parameter.
    """
    rng = np.random.default_rng(seed)
    omega = rng.normal(omega_mean, omega_std, size=N)

    if theta0 is None:
        theta0 = rng.uniform(-np.pi, np.pi, size=N)

    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    sol = solve_ivp(
        fun=kuramoto_rhs,
        t_span=t_span,
        y0=theta0,
        args=(omega, K),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )

    t = sol.t                     # (T,)
    theta = sol.y.T               # (T, N)
    r = order_parameter(theta)    # (T,)

    return t, theta, omega, r


# ──────────────────────────────────────────────
#  보조 함수
# ──────────────────────────────────────────────

def order_parameter(theta: np.ndarray) -> np.ndarray:
    """동기화 정도(order parameter) r(t) 계산.

    Parameters
    ----------
    theta : ndarray, shape (T, N)
        위상 궤적.

    Returns
    -------
    r : ndarray, shape (T,)
        각 시각의 order parameter (0 ≤ r ≤ 1).
    """
    return np.abs(np.mean(np.exp(1j * theta), axis=1))


def mean_phase(theta: np.ndarray) -> np.ndarray:
    """평균 위상 ψ(t) 계산.

    Returns
    -------
    psi : ndarray, shape (T,)  [rad]
    """
    z = np.mean(np.exp(1j * theta), axis=1)
    return np.angle(z)


# ══════════════════════════════════════════════
#  Explosive Synchronization (ES) 확장
# ══════════════════════════════════════════════
#
# 핵심 수식 (Gómez-Gardeñes et al., PRL 2011):
#
#   dθ_i/dt = ω_i + K · Σ_j A_ij · sin(θ_j − θ_i)
#
#   - A_ij: 인접 행렬 (BA scale-free 네트워크)
#   - ω_i = k_i (차수-주파수 상관)  ← ES의 핵심 조건
#
# 전결합 쿠라모토와의 차이:
#   전결합: 2차 전이 (연속, r ~ sqrt(K - K_c))
#   ES:     1차 전이 (비연속) + 히스테리시스
#            → 정방향 스윕 K_f  ≠  역방향 스윕 K_b  (K_f > K_b)
# ──────────────────────────────────────────────


def kuramoto_network_rhs(t, theta, omega, K, A):
    """네트워크 쿠라모토 ODE 우변.

    Parameters
    ----------
    t : float
    theta : ndarray, shape (N,)
    omega : ndarray, shape (N,)  — 고유 진동수 (ES: ω_i = k_i)
    K : float                   — 결합 강도
    A : ndarray, shape (N, N)   — 인접 행렬

    Returns
    -------
    dtheta_dt : ndarray, shape (N,)
    """
    theta = np.asarray(theta)
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]   # diff[i,j] = θ_j − θ_i
    # A_ij * sin(θ_j − θ_i) 의 행 합산
    coupling = K * np.sum(A * np.sin(diff), axis=1)
    return omega + coupling


def solve_kuramoto_network(
    A: np.ndarray,
    omega: np.ndarray,
    K: float,
    t_span: tuple = (0.0, 100.0),
    dt: float = 0.1,
    theta0: np.ndarray = None,
    seed: int = 42,
):
    """네트워크 쿠라모토 모델 적분 (ES 포함).

    Parameters
    ----------
    A : ndarray, shape (N, N)   인접 행렬
    omega : ndarray, shape (N,) 고유 진동수
    K : float                   결합 강도
    t_span : (t0, tf)
    dt : float                  저장 간격
    theta0 : ndarray, optional  초기 위상 (None → 균일 랜덤)
    seed : int

    Returns
    -------
    t      : ndarray (T,)
    theta  : ndarray (T, N)
    r      : ndarray (T,)   order parameter
    """
    N = len(omega)
    rng = np.random.default_rng(seed)
    if theta0 is None:
        theta0 = rng.uniform(-np.pi, np.pi, size=N)

    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    sol = solve_ivp(
        fun=kuramoto_network_rhs,
        t_span=t_span,
        y0=theta0,
        args=(omega, K, A),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-7,
    )

    t = sol.t
    theta = sol.y.T
    r = order_parameter(theta)
    return t, theta, r


def sweep_K_bifurcation(
    A: np.ndarray,
    omega: np.ndarray,
    K_values: np.ndarray,
    t_settle: float = 80.0,
    t_measure: float = 20.0,
    dt: float = 0.1,
    forward: bool = True,
    seed: int = 42,
):
    """K를 순차적으로 변화시키며 정상 상태 r을 측정 (히스테리시스 스윕).

    Parameters
    ----------
    A : ndarray, shape (N, N)
    omega : ndarray, shape (N,)
    K_values : ndarray           검사할 K 값 배열 (오름차순 또는 내림차순)
    t_settle : float             각 K에서 과도 응답 제거 시간 [s]
    t_measure : float            정상 상태 r 평균 내는 구간 [s]
    dt : float
    forward : bool               True=정방향(K 증가), False=역방향(K 감소)
    seed : int

    Returns
    -------
    K_out   : ndarray   K 배열 (입력과 동일 순서)
    r_mean  : ndarray   각 K에서의 시간 평균 r
    r_std   : ndarray   각 K에서의 r 표준편차 (CSD 신호)
    r_ac    : ndarray   각 K에서의 r autocorrelation lag-1
    """
    N = len(omega)
    rng = np.random.default_rng(seed)

    if forward:
        # 정방향: 완전 비동기 초기 상태
        theta_cur = rng.uniform(-np.pi, np.pi, size=N)
    else:
        # 역방향: 완전 동기 초기 상태
        theta_cur = np.zeros(N)

    K_list, r_mean_list, r_std_list, r_ac_list = [], [], [], []

    for K in K_values:
        # 과도 응답 제거
        t_s, theta_s, r_s = solve_kuramoto_network(
            A, omega, K,
            t_span=(0, t_settle),
            dt=dt,
            theta0=theta_cur.copy(),
        )
        theta_cur = theta_s[-1].copy()   # 다음 K의 초기 상태로 연속

        # 정상 상태 측정
        t_m, theta_m, r_m = solve_kuramoto_network(
            A, omega, K,
            t_span=(0, t_measure),
            dt=dt,
            theta0=theta_cur.copy(),
        )
        theta_cur = theta_m[-1].copy()

        # 통계
        r_mean = r_m.mean()
        r_std  = r_m.std()
        if len(r_m) > 2:
            r_ac = float(np.corrcoef(r_m[:-1], r_m[1:])[0, 1])
        else:
            r_ac = np.nan

        K_list.append(K)
        r_mean_list.append(r_mean)
        r_std_list.append(r_std)
        r_ac_list.append(r_ac)

    return (np.array(K_list),
            np.array(r_mean_list),
            np.array(r_std_list),
            np.array(r_ac_list))


# ──────────────────────────────────────────────
#  간단 테스트
# ──────────────────────────────────────────────
