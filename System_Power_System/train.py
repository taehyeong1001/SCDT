"""Train Power System using the final 60,000-sample training protocol."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import runtime as rt
from common.training import train_main
import numpy as np


def fit():
    _, model, weights, trajectories, _ = rt.power_load()
    n = model.n_reservoir
    gram, cross = np.zeros((n, n)), np.zeros((4, n))
    states, targets = np.empty((2048, n)), np.empty((2048, 4))
    used, count = 0, 0
    for trajectory, feature in zip(trajectories.values(), weights['training_features']):
        state, bias = np.zeros(n), model._compute_bias(feature)
        for t in range(59999):
            state = model._step(state, trajectory[t], bias)
            if t < 100:
                continue
            states[used], targets[used] = state, trajectory[t + 1]
            used += 1
            if used == len(states):
                gram += states.T @ states
                cross += targets.T @ states
                count += used
                used = 0
    if used:
        gram += states[:used].T @ states[:used]
        cross += targets[:used].T @ states[:used]
        count += used
    assert count == 299495
    fitted = np.linalg.solve(gram + model.reg * np.eye(n), cross.T).T
    return fitted, model.W_out, dict(washout=100, steps=60000, conditions=5, effective_pairs=count)


if __name__ == '__main__':
    train_main('power', fit)
