# -*- coding: utf-8 -*-
"""Power ODE helpers, statistical reservoir and autonomous rollout API.

Final saved weights and constant-conditioning evaluation are loaded by
common/runtime.py. Generic constructor defaults are not the paper settings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pickle
from solver.voltage_model import VoltageCollapseModel
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------
# Utilities
# ---------------------------------------------

def calculate_autocorrelation(data, lag=1):
    data = np.array(data)
    mean = np.mean(data)
    c0   = np.sum((data - mean) ** 2) / len(data)
    if lag >= len(data) or c0 == 0:
        return 0.0
    c_lag = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / (len(data) - lag)
    return c_lag / c0


# ---------------------------------------------
# StatisticalReservoir  (FoodChain bias ���)
# ---------------------------------------------

class StatisticalReservoir:
    """
    FoodChain ���� ���:
      reservoir update: r = (1-a)*r + a*tanh(A*r + W_in*u + W_stat*feat)
      �Է� u  : [dm, w, d, V]  4D  (���� ���͸�)
      feat    : normalize([std, ac]) -> ���� bias ���ͷ� ����
    """
    def __init__(self, n_reservoir=800, spectral_radius=1.6, sparsity=0.6875,
                 input_scaling=2.1, stat_scaling=1.0, leakage=1.0, reg=1e-4, seed=42):
        self.n_reservoir    = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity       = sparsity
        self.input_scaling  = input_scaling
        self.stat_scaling   = stat_scaling
        self.leakage        = leakage
        self.reg            = reg
        self.W_out          = None
        self.last_state     = None

        np.random.seed(seed)
        W    = np.random.randn(n_reservoir, n_reservoir)
        mask = np.random.rand(n_reservoir, n_reservoir) > sparsity
        W    = W * mask
        ev   = np.linalg.eigvals(W)
        self.W_reservoir = W * (spectral_radius / np.max(np.abs(ev)))

        # �Է� ����ġ: 4D (���� ���� [dm,w,d,V])
        self.W_in   = np.random.randn(n_reservoir, 4) * input_scaling
        # ��� ä�� ����ġ: 2D (normalized [std, ac]) -> bias ����  (stat_scaling���� ũ�� ����)
        self.W_stat = np.random.randn(n_reservoir, 2) * stat_scaling

    def _compute_bias(self, feat_norm):
        """feat_norm: (2,) normalized [std, ac] -> (n_res,) bias"""
        return self.W_stat @ np.array(feat_norm)

    def _step(self, state, u, bias):
        """u: (4,), bias: (n_res,)"""
        pre = self.W_reservoir @ state + self.W_in @ u + bias
        return (1 - self.leakage) * state + self.leakage * np.tanh(pre)

    def fit(self, X_list, y_list, feat_norm_list, washout=100):
        """
        X_list       : list of (n_i, 4) - �� Q1�� ���� �ð迭 (�Է�)
        y_list       : list of (n_i, 4) - �� Q1�� ���� ���� (Ÿ��)
        feat_norm_list: list of (2,)    - �� Q1�� normalize([std, ac])
        """
        states_all = []
        y_all      = []
        state      = np.zeros(self.n_reservoir)

        for X_i, y_i, feat_i in zip(X_list, y_list, feat_norm_list):
            bias  = self._compute_bias(feat_i)
            state = np.zeros(self.n_reservoir)
            for t in range(len(X_i)):
                state = self._step(state, X_i[t], bias)
                if t >= washout:
                    states_all.append(state.copy())
                    y_all.append(y_i[t])

        S = np.array(states_all)   # (N, n_res)
        Y = np.array(y_all)        # (N, 4)

        A_inv      = np.linalg.inv(S.T @ S + self.reg * np.eye(self.n_reservoir))
        self.W_out = Y.T @ S @ A_inv   # (4, n_res)
        self.last_state = state

    def run_warmup(self, X, feat_norm, reset_state=True):
        """X: (n, 4) - warmup���� reservoir ���� �ʱ�ȭ"""
        bias  = self._compute_bias(feat_norm)
        state = np.zeros(self.n_reservoir) if reset_state else self.last_state.copy()
        for u in X:
            state = self._step(state, u, bias)
        self.last_state = state

    def predict_one(self, u, feat_norm):
        """���� ���� ����. last_state ������Ʈ. ��ȯ: (4,)"""
        bias            = self._compute_bias(feat_norm)
        state           = self._step(self.last_state, u, bias)
        self.last_state = state
        return self.W_out @ state   # (4,)


# ---------------------------------------------
# Safe Basin Initial Condition Sampler
# ---------------------------------------------

def get_safe_initial_condition(seed=42):
    """IC: optimizer와 동일한 narrower 범위 (훈련 데이터 일치용)"""
    rng = np.random.default_rng(seed)
    return np.array([
        rng.uniform(0.17, 0.25),   # deltam
        rng.uniform(0.00, 0.05),   # omega
        rng.uniform(0.05, 0.10),   # delta
        rng.uniform(0.83, 0.87),   # V
    ])


def get_collapse_initial_condition(seed=42):
    """IC: delayed collapse 분석에서 확인된 범위 (a2/b2 collapse용)
    V=[0.70,0.87], deltam=[0.17,0.42]  (find_delayed_collapse_ic.py 결과 기반)
    """
    rng = np.random.default_rng(seed)
    return np.array([
        rng.uniform(0.17, 0.42),   # deltam (wider)
        0.025,                     # omega  (fixed)
        0.10,                      # delta  (fixed)
        rng.uniform(0.70, 0.87),   # V (lower bound 포함)
    ])


# ---------------------------------------------
# Simulation helpers
# ---------------------------------------------

def simulate_stable(Q1, n_steps, dt=0.05, t_transient=500, seed=42):
    """500s Ʈ������Ʈ ���� �� n_steps ���� ���� (4D)."""
    m = VoltageCollapseModel(Q1=Q1)
    for retry in range(30):
        try:
            x0       = get_safe_initial_condition(seed=seed + retry)
            t_total  = t_transient + n_steps * dt
            _, x_all = m.simulate(x0, (0, t_total), dt=dt, verbose=False,
                                   method='RK45', max_step=0.1, rtol=1e-4, atol=1e-6)
            n_cut    = int(t_transient / dt)
            if len(x_all) < n_cut + n_steps:
                continue
            x = x_all[n_cut: n_cut + n_steps]
            if np.min(x[:, 3]) < 0.5 or np.any(np.isnan(x)):
                continue
            return x
        except Exception:
            continue
    return None


def simulate_gt(Q1, total_steps, dt=0.05, t_transient=500, seed=42):
    """GT �÷Կ� V �ð迭.
    Q1 < Q1c (stable): 500s Ʈ������Ʈ ���� �� ���� ����.
    Q1 > Q1c (collapse): Ʈ������Ʈ ���� �ٷ� �ù� (�ر��� t=10~20s�� �߻��ϹǷ�).
    """
    Q1c = 2.9898256
    m   = VoltageCollapseModel(Q1=Q1)

    if Q1 < Q1c:
        # ���� Stable: Ʈ������Ʈ �� ���� ���� ����
        for retry in range(30):
            x0       = get_safe_initial_condition(seed=seed + retry)
            sim_time = t_transient + total_steps * dt
            _, x_all = m.simulate(x0, (0, sim_time), dt=dt, verbose=False,
                                   method='RK45', max_step=0.1, rtol=1e-4, atol=1e-6)
            n_cut = int(t_transient / dt)
            if len(x_all) < n_cut + 50:
                continue
            v = x_all[n_cut:, 3]
            if np.any(np.isnan(v[:50])) or np.min(v) < 0.3 or v[-1] < 0.5:
                continue
            if len(v) < total_steps:
                v = np.concatenate([v, np.full(total_steps - len(v), np.nan)])
            return v[:total_steps], x0
        return None, None

    else:
        # Collapse: delayed collapse IC 우선 탐색 (5s < tc <= 80s)
        T_FAST_GT = 5.0
        T_SLOW_GT = 80.0
        best_v, best_x0 = None, None

        for retry in range(60):
            x0 = get_collapse_initial_condition(seed=seed + retry)
            try:
                _, x_all = m.simulate(x0, (0, total_steps * dt), dt=dt, verbose=False,
                                       method='RK45', max_step=0.1, rtol=1e-4, atol=1e-6)
            except Exception:
                continue
            if len(x_all) < 50:
                continue
            v = x_all[:, 3]
            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                continue
            # 붕괴 시각 계산
            idx = np.argwhere(v < 0.60)
            tc = float(np.arange(len(v))[idx[0, 0]] * dt) if len(idx) else total_steps * dt + 1
            if len(v) < total_steps:
                v = np.concatenate([v, np.full(total_steps - len(v), np.nan)])
            if T_FAST_GT < tc <= T_SLOW_GT:
                # delayed collapse IC 발견 -> 즉시 반환
                print(f'  GT collapse IC: dm={x0[0]:.3f}, V={x0[3]:.3f}  tc={tc:.1f}s (delayed)')
                return v[:total_steps], x0
            elif best_v is None and tc <= total_steps * dt:
                # delayed는 아니지만 붕괴는 하는 IC를 fallback으로 저장
                best_v, best_x0 = v[:total_steps], x0

        if best_v is not None:
            print(f'  GT collapse IC (fallback): dm={best_x0[0]:.3f}, V={best_x0[3]:.3f}')
            return best_v, best_x0
        # 최후 fallback
        x0 = get_collapse_initial_condition(seed=seed)
        _, x_all = m.simulate(x0, (0, total_steps * dt), dt=dt, verbose=False,
                               method='RK45', max_step=0.1, rtol=1e-4, atol=1e-6)
        v = x_all[:, 3]
        if len(v) < total_steps:
            v = np.concatenate([v, np.full(total_steps - len(v), np.nan)])
        return v[:total_steps], x0


# ---------------------------------------------
# Training  (FoodChain bias ���)
# ---------------------------------------------



# ---------------------------------------------
# RC autoregressive prediction  (bias ���)
# ---------------------------------------------

def rc_predict_trajectory(rc, Q1, Q1_train_list, feat_min, feat_max, poly_std, poly_ac,
                           train_trajs=None, n_train=10000, feat_norm_dict=None,
                           feat_stable=None, feat_transition_steps=0,
                           warmup_Q1=None,
                           warmup_steps=100, pred_steps=2000,
                           dt=0.05, seed=42):
    """
    optimizer와 동일한 warmup/cur 로직:
      warmup: traj[n_train-100 : n_train, :]
      cur   : traj[n_train-1, :]
    feat 점진적 전환 (collapse용):
      feat_stable         : 시작 feat (훈련 최대 Q1의 정확한 feat)
      feat_transition_steps: 0 이면 전환 없음. N이면 step 0~N 동안 stable→OOD 선형 보간
      warmup_Q1           : warmup 시 사용할 Q1 (None이면 가장 가까운 훈련 Q1)
      → stable Q1 feat로 warmup → t=0 근방에서 진동 유지 → 이후 OOD bias로 붕괴 유도
    """
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0

    def minmax(std_v, ac_v):
        return (np.array([std_v, ac_v]) - feat_min) / feat_range

    # 훈련 Q1이면 정확한 feat 사용, OOD이면 poly 외삽
    tol = 1e-8
    if feat_norm_dict is not None and any(abs(Q1 - q) < tol for q in Q1_train_list):
        q_match = min(Q1_train_list, key=lambda q: abs(q - Q1))
        feat_norm = feat_norm_dict[q_match]
        std_extrap = float(poly_std(Q1))   # 표시용
        ac_extrap  = float(poly_ac(Q1))
    else:
        std_extrap = float(poly_std(Q1))
        ac_extrap  = float(poly_ac(Q1))
        feat_norm  = minmax(std_extrap, ac_extrap)

    Q1c_train = max(Q1_train_list)
    tag = "in range" if Q1 <= Q1c_train else "OOD (extrapolated)"
    print(f"    Q1={Q1:.6f}  std={std_extrap:.6f}  ac={ac_extrap:.4f}"
          f"  norm=[{feat_norm[0]:+.3f},{feat_norm[1]:+.3f}]  ({tag})")

    # Warmup Q1 결정: warmup_Q1이 지정되면 해당 Q1 사용, 아니면 가장 가까운 훈련 Q1
    if warmup_Q1 is not None and warmup_Q1 in Q1_train_list:
        Q1_wu = warmup_Q1
    else:
        Q1_wu = min(Q1_train_list, key=lambda q: abs(q - min(max(Q1_train_list), Q1)))
    if train_trajs is not None and Q1_wu in train_trajs:
        traj_wu = train_trajs[Q1_wu]             # (n_train+n_val, 4) 전체 궤적
        # optimizer 방식: traj[n_train-100 : n_train, :]
        x_wu = traj_wu[n_train - warmup_steps : n_train, :]
        cur  = traj_wu[n_train - 1, :].copy()   # optimizer 방식: cur = traj[n_train-1]
        if len(x_wu) < warmup_steps:
            x_wu = np.tile(traj_wu[0], (warmup_steps, 1))
            cur  = traj_wu[min(n_train-1, len(traj_wu)-1), :].copy()
    else:
        # fallback: 짧은 시뮬 (500s transient 포함)
        x0_wu = get_safe_initial_condition(seed=seed)
        m_wu  = VoltageCollapseModel(Q1=Q1_wu)
        try:
            _, x_wu_all = m_wu.simulate(
                x0_wu, (0, 500 + (warmup_steps + 1) * dt),
                dt=dt, verbose=False, method='RK45', max_step=0.1, rtol=1e-4, atol=1e-6)
            n_cut = int(500 / dt)
            x_wu  = (x_wu_all[n_cut: n_cut + warmup_steps, :]
                     if len(x_wu_all) >= n_cut + warmup_steps
                     else np.tile(x0_wu, (warmup_steps, 1)))
            cur = x_wu[-1, :].copy()
        except Exception:
            x_wu = np.tile(x0_wu, (warmup_steps, 1))
            cur  = x_wu[-1, :].copy()

    # Reservoir warmup: feat_stable이 있으면 그걸로 warmup (stable attractor 근방으로 초기화)
    warmup_feat = feat_stable if feat_stable is not None else feat_norm
    rc.run_warmup(x_wu, warmup_feat, reset_state=True)
    states_pred = []

    # 훈련 데이터 범위 기반 4D clip (발산 방지)
    STATE_MIN = np.array([-0.10, -1.50, -0.30, 0.01])
    STATE_MAX = np.array([ 0.70,  1.20,  0.40, 1.10])

    for step in range(pred_steps):
        # feat 전환 전략 (collapse 예측 시):
        #   step 0 ~ feat_transition_steps-1 : stable feat 고정 → 진동 유지
        #   step feat_transition_steps ~ +50  : stable→OOD 선형 전환 (빠른 ramp)
        #   step 이후                          : OOD feat 고정 → 붕괴
        RAMP = 50   # 전환 ramp 구간 (50 steps = 2.5s)
        if feat_stable is not None and feat_transition_steps > 0:
            if step < feat_transition_steps:
                feat_cur = feat_stable
            elif step < feat_transition_steps + RAMP:
                alpha    = (step - feat_transition_steps) / RAMP
                feat_cur = (1.0 - alpha) * feat_stable + alpha * feat_norm
            else:
                feat_cur = feat_norm
        else:
            feat_cur = feat_norm
        p = rc.predict_one(cur, feat_cur)
        p = np.clip(p, STATE_MIN, STATE_MAX)
        states_pred.append(p.copy())
        cur = p

        if (step + 1) % 500 == 0 or (step + 1) in (1, 10, 50, 200, 300, 400):
            print(f"      step={step+1:4d}  V={p[3]:.4f}  feat=[{feat_cur[0]:.3f},{feat_cur[1]:.3f}]")

    return x_wu[:, 3], np.array(states_pred)[:, 3]


# ---------------------------------------------
# Figure 7 Style Plot
# ---------------------------------------------



# ---------------------------------------------
# Main
# ---------------------------------------------
