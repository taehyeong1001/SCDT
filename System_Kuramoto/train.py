"""Train the final r-only Kuramoto readout with 2,000 reservoir nodes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import runtime as rt
from common.training import train_main
import numpy as np
from scipy.linalg import eigh


def fit():
    model, arrays, _, _ = rt.kuramoto_load()
    sequences, features = arrays['train_sequences'], arrays['train_features']
    state = np.zeros((model.n, len(sequences)))
    bias = model.w_stat @ features.T
    x, y = [], []
    for t in range(sequences.shape[1] - 1):
        current = sequences[:, t, 0][None, :]
        state = (1 - model.leakage) * state + model.leakage * np.tanh(model.a @ state + model.w_in @ current + bias)
        if t >= 80:
            x.append(np.vstack([state, current, np.ones((1, len(sequences)))]).T)
            y.append(sequences[:, t + 1, 0])
    x, y = np.concatenate(x), np.concatenate(y)
    gram, rhs = x.T @ x, x.T @ y
    eigenvalues, vectors = eigh(gram, check_finite=False)
    fitted = (vectors @ ((vectors.T @ rhs) / (eigenvalues + model.reg)))[None, :]
    residual = np.linalg.norm((gram + model.reg * np.eye(len(gram))) @ fitted[0] - rhs) / np.linalg.norm(rhs)
    assert residual < 1e-8
    return fitted, model.w_out, dict(washout=80, steps=2301, conditions=5,
                                    effective_pairs=len(y), ridge_normal_equation_residual=float(residual))


if __name__ == '__main__':
    train_main('kuramoto', fit)
