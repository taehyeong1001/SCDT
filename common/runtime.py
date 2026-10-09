"""Frozen-model inference, with the original update and warm-up protocols."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def food_load():
    module = load_module('food_runtime_impl', ROOT / 'System_Food_Chain/experiment.py')
    with np.load(ROOT / 'models/food_chain/checkpoint.npz', allow_pickle=False) as z:
        setup = {name: z[name].copy() for name in z.files}
    params = json.loads((ROOT / 'models/food_chain/params.json').read_text())
    model = module.StatRC.__new__(module.StatRC)
    for name in ('W_in', 'W_stat', 'W_out', 's_vec', 'b_vec'):
        setattr(model, name, setup[name])
    model.A = csr_matrix((setup['A_data'], setup['A_indices'], setup['A_indptr']),
                         shape=tuple(setup['A_shape']))
    model.p = params
    return module, model, setup


def food_feature(setup, sigma, entropy=None):
    if entropy is None:
        entropy = np.polyval(np.polyfit(setup['stats'][:, 0], setup['stats'][:, 1], 1), sigma)
    feature = (np.array([sigma, entropy]) - setup['scaler_mean']) / setup['scaler_scale']
    return feature, float(entropy)


def food_predict(loaded, source, start, sigma, steps, entropy=None, selected_post=False):
    module, model, setup = loaded
    target, entropy = food_feature(setup, sigma, entropy)
    warmup = setup['sequences'][source, start:start + 500]
    if len(warmup) != 500:
        raise ValueError('Food warm-up must contain exactly 500 training samples')
    initial = setup['normalized'][-1] if selected_post else target
    return module.rc_predict_delayed(model, warmup, initial, target,
                                    stable_steps=500 if selected_post else 0,
                                    ramp_steps=50 if selected_post else 0,
                                    total_steps=steps)[:, 2]


def food_status(signal, tail=500):
    finite = bool(np.isfinite(signal).all())
    average = float(np.mean(signal[-tail:]))
    return dict(finite=finite, tail_mean=average, survived=finite and average > .05)


def power_load():
    folder = ROOT / 'models/voltage_60000'
    protocol = json.loads((folder / 'protocol.json').read_text())
    module = load_module('power_runtime_impl', ROOT / 'System_Power_System/Reservoir/trajectory_prediction.py')
    with np.load(folder / 'checkpoint.npz', allow_pickle=False) as z:
        weights = {name: z[name].copy(order='K') for name in z.files}
    with np.load(folder / 'refitted_readout.npz') as z:
        weights['W_out'] = z['W_out'].copy(order='K')
    rc = module.StatisticalReservoir.__new__(module.StatisticalReservoir)
    rc.__dict__.update(protocol['reservoir_attributes'])
    for name in ('W_reservoir', 'W_in', 'W_stat', 'W_out'):
        setattr(rc, name, weights[name])
    rc.last_state = None
    trajectories = {}
    for i, q in enumerate(protocol['train_Q1']):
        with np.load(folder / f'train_{i}.npz') as z:
            trajectories[q] = z['retained_state'].copy()
    points = pd.read_csv(ROOT / 'data/voltage_consistent_final/selected_points.csv')
    train = points[points.role.eq('train')]
    ac_fit = np.polyfit(train.stat_1, train.stat_2, 1)
    return module, rc, weights, trajectories, ac_fit


def power_predict(loaded, query_q, sigma, end, steps, ac=None):
    module, rc, weights, trajectories, ac_fit = loaded
    if ac is None:
        ac = float(np.polyval(ac_fit, sigma))
    source_q = min(trajectories, key=lambda q: abs(q - min(max(trajectories), query_q)))
    if end < 100 or end > len(trajectories[source_q]):
        raise ValueError('Power warm-up endpoint is outside the training record')
    _, v = module.rc_predict_trajectory(
        rc, query_q, list(trajectories), weights['feature_min'], weights['feature_max'],
        lambda _: sigma, lambda _: ac, train_trajs=trajectories, n_train=end,
        feat_norm_dict=None, warmup_Q1=source_q, warmup_steps=100, pred_steps=steps,
        dt=.05, seed=42, feat_stable=None, feat_transition_steps=0)
    return v


def power_status(module, signal):
    finite = bool(np.isfinite(signal).all())
    events = np.flatnonzero(signal <= .30)
    survived = finite and not len(events)
    return dict(finite=finite, survived=bool(survived), min_voltage=float(np.nanmin(signal)),
                first_threshold_step=int(events[0]) if len(events) else -1,
                reproduced_std=float(np.std(signal)) if survived else float('nan'),
                reproduced_ac=float(module.calculate_autocorrelation(signal)) if survived else float('nan'))


def kuramoto_load():
    with np.load(ROOT / 'models/kuramoto/model_checkpoint.npz', allow_pickle=False) as z:
        arrays = {name: z[name].copy(order='K') for name in z.files}
    model = SimpleNamespace(a=arrays['a'], w_in=arrays['w_in'], w_stat=arrays['w_stat'],
                            w_out=arrays['w_out'], leakage=float(arrays['leakage'][0]),
                            reg=float(arrays['reg'][0]), n=arrays['a'].shape[0])
    training = pd.read_csv(ROOT / 'models/kuramoto/training_coordinates.csv')
    params = json.loads((ROOT / 'models/kuramoto/params.json').read_text())
    return model, arrays, training, params


def kuramoto_target(loaded, sigma):
    _, arrays, training, _ = loaded
    int_fit = np.polyfit(training.stat_1, training.stat_2, 1)
    out = float((sigma - int_fit[1]) / int_fit[0])
    k_fit = np.polyfit(training.K, training.stat_1, 1)
    query = float((out - k_fit[1]) / k_fit[0])
    source = int(np.argmin(abs(training.K.to_numpy() - query)))
    target = (np.array([out, sigma]) - arrays['feature_mean']) / arrays['feature_std']
    return target, source, out, query


def kuramoto_windows(sequence):
    starts = [int(round((len(sequence) - 60) * (i + 1) / 21)) for i in range(20)]
    return np.array([sequence[s:s + 60] for s in starts])


def kuramoto_predict(loaded, sigma, steps=10000, window_indices=None):
    model, arrays, training, params = loaded
    feature, source, out, query = kuramoto_target(loaded, sigma)
    warmups = kuramoto_windows(arrays['train_sequences'][source])
    if window_indices is not None:
        warmups = warmups[window_indices]
    members = len(warmups)
    state = np.zeros((model.n, members))
    bias = (model.w_stat @ arrays['train_features'][source])[:, None]
    for t in range(60):
        current = warmups[:, t, 0][None, :]
        state = (1 - model.leakage) * state + model.leakage * np.tanh(model.a @ state + model.w_in @ current + bias)
    current = warmups[:, -1, 0][None, :]
    bias = (model.w_stat @ feature)[:, None]
    ones = np.ones((1, members))
    predictions = np.empty((members, steps))
    for t in range(steps):
        state = (1 - model.leakage) * state + model.leakage * np.tanh(model.a @ state + model.w_in @ current + bias)
        current = np.clip(model.w_out @ np.vstack([state, current, ones]),
                          -params['clip_value'], params['clip_value'])
        predictions[:, t] = current[0]
    values = np.clip(predictions * arrays['sequence_std'][0] + arrays['sequence_mean'][0], 0, 1)
    return values, dict(S_out=out, original_query_parameter=query, warmup_source_K=float(training.K.iloc[source]))


def kuramoto_status(values, tail=3000):
    tail_values = values[:, -tail:]
    dwell = np.mean(tail_values >= .5, axis=1)
    flags = np.isfinite(values).all(axis=1) & (dwell >= .8)
    return dwell, flags


def crossing(x, rate, increasing=False):
    x, rate = np.asarray(x), np.asarray(rate)
    for i in range(len(x) - 1):
        if rate[i] == .5:
            return float(x[i])
        bracket = rate[i] < .5 <= rate[i + 1] if increasing else rate[i] > .5 >= rate[i + 1]
        if bracket:
            return float(x[i] + (.5 - rate[i]) * (x[i + 1] - x[i]) / (rate[i + 1] - rate[i]))
    if rate[-1] == .5:
        return float(x[-1])
    raise ValueError('No 50% crossing in the evaluated range')
