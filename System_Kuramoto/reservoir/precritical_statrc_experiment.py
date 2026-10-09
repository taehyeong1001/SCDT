"""Generic scaler and statistical reservoir API retained from the experiment.

Final r-only batching, fitted settings and training-window protocols are in
common/runtime.py and system-specific train.py; this file is not a standalone experiment.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# solver import
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT_DIR)

from solver.kuramoto_model import solve_kuramoto_network
from solver.network_utils import (
    assign_frequencies_es_normalized,
    get_adjacency_matrix,
    make_ba_network,
)






class StatReservoir:
    """FoodChain bias 형태의 간단 StatRC."""

    def __init__(
        self,
        n_reservoir: int,
        input_dim: int,
        stat_dim: int,
        output_dim: int,
        spectral_radius: float,
        sparsity: float,
        input_scaling: float,
        stat_scaling: float,
        leakage: float,
        reg: float,
        seed: int = 42,
    ):
        self.n = n_reservoir
        self.input_dim = input_dim
        self.stat_dim = stat_dim
        self.output_dim = output_dim
        self.leakage = leakage
        self.reg = reg

        rng = np.random.default_rng(seed)

        w = rng.standard_normal((n_reservoir, n_reservoir))
        mask = rng.random((n_reservoir, n_reservoir)) > sparsity
        w[mask] = 0.0
        eigvals = np.linalg.eigvals(w)
        radius = np.max(np.abs(eigvals))
        radius = radius if radius > 1e-12 else 1.0
        self.a = w * (spectral_radius / radius)

        self.w_in = rng.standard_normal((n_reservoir, input_dim)) * input_scaling
        self.w_stat = rng.standard_normal((n_reservoir, stat_dim)) * stat_scaling
        self.w_out = np.zeros((output_dim, n_reservoir + input_dim + 1))

    def _step(self, r: np.ndarray, u: np.ndarray, feat: np.ndarray) -> np.ndarray:
        pre = self.a @ r + self.w_in @ u + self.w_stat @ feat
        return (1.0 - self.leakage) * r + self.leakage * np.tanh(pre)

    def _collect_states(self, u_seq: np.ndarray, feat: np.ndarray, washout: int) -> tuple[np.ndarray, np.ndarray]:
        t_steps = len(u_seq)
        r = np.zeros(self.n)
        states = []
        for t in range(t_steps - 1):
            r = self._step(r, u_seq[t], feat)
            if t >= washout:
                ext = np.concatenate([r, u_seq[t], np.array([1.0])])
                states.append(ext)
        x = np.asarray(states)  # (T-washout-1, n+input+1)
        y = u_seq[washout + 1 : t_steps]  # next-step target
        return x, y

    def fit(self, sequences: list[np.ndarray], feats: list[np.ndarray], washout: int = 100) -> None:
        x_list = []
        y_list = []
        for u_seq, feat in zip(sequences, feats):
            x, y = self._collect_states(u_seq, feat, washout)
            x_list.append(x)
            y_list.append(y)

        x_all = np.vstack(x_list)
        y_all = np.vstack(y_list)

        xx = x_all.T @ x_all
        reg_i = self.reg * np.eye(xx.shape[0])
        self.w_out = (np.linalg.solve(xx + reg_i, x_all.T @ y_all)).T

    def one_step_predict(self, u_seq: np.ndarray, feat: np.ndarray, washout: int = 100) -> np.ndarray:
        r = np.zeros(self.n)
        preds = []
        targets = []
        for t in range(len(u_seq) - 1):
            r = self._step(r, u_seq[t], feat)
            if t >= washout:
                ext = np.concatenate([r, u_seq[t], np.array([1.0])])
                pred = self.w_out @ ext
                preds.append(pred)
                targets.append(u_seq[t + 1])
        return np.asarray(preds), np.asarray(targets)

    def rollout(
        self,
        u_seed: np.ndarray,
        feat: np.ndarray,
        n_steps: int,
        warmup: int = 50,
        clip_value: float = 5.0,
    ) -> np.ndarray:
        """teacher forcing warmup 후 autonomous rollout."""
        r = np.zeros(self.n)
        u_cur = u_seed[0].copy()

        for t in range(min(warmup, len(u_seed))):
            u_cur = u_seed[t]
            r = self._step(r, u_cur, feat)

        out = []
        for _ in range(n_steps):
            r = self._step(r, u_cur, feat)
            ext = np.concatenate([r, u_cur, np.array([1.0])])
            u_cur = self.w_out @ ext
            # 폭주 억제를 위해 표준화 공간에서 clipping
            u_cur = np.clip(u_cur, -clip_value, clip_value)
            out.append(u_cur.copy())
        return np.asarray(out)















