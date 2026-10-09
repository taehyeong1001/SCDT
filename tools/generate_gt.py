"""Regenerate GT baselines with the frozen initial conditions and solver protocol."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'scdt-mpl'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import runtime as rt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def food(args):
    module = rt.load_module('food_gt_equations', ROOT / 'common/gt_baselines/food_legacy.py')
    frame = pd.read_csv(ROOT / 'data/gt_baselines/food_gt_runs.csv')
    requested = pd.read_csv(ROOT / 'data/food_rate_single_rule_20261001/gt_rate.csv').actual_parameter.to_numpy()
    rows = []
    for i, k in enumerate(requested):
        if args.condition is not None and i not in args.condition:
            continue
        group = frame[np.isclose(frame.K, k, atol=1e-12, rtol=0)]
        if args.members:
            group = group.iloc[:args.members]
        for record in group.itertuples():
            initial = [record.initial_R, record.initial_C, record.initial_P]
            solution = solve_ivp(module.food_chain_eq, (0, 2500), initial, args=(float(k),),
                                 method='RK45', t_eval=np.arange(25001) / 10, rtol=1e-8, atol=1e-10)
            complete = solution.success and solution.y.shape[1] == 25001 and np.isfinite(solution.y).all()
            signal = solution.y[2, 5000:15000:10] if complete else np.full(1000, np.nan)
            np.savez_compressed(args.output / f'condition_{i:03d}_member_{record.member:02d}.npz',
                                signal=signal, initial=initial, parameter=k)
            rows.append(dict(condition_index=i, K=k, member=record.member, complete=bool(complete),
                             **rt.food_status(signal)))
        print(f'Food GT {i} complete', flush=True)
    return rows


def power(args):
    module = rt.load_module('power_gt_equations', ROOT / 'common/gt_baselines/power_model.py')
    frame = pd.read_csv(ROOT / 'data/gt_baselines/power_gt_runs.csv')
    requested = pd.read_csv(ROOT / 'data/matched_rate_raw_axes_20261001/power/gt_rate.csv').actual_parameter.to_numpy()
    rows = []
    for i, q in enumerate(requested):
        if args.condition is not None and i not in args.condition:
            continue
        group = frame[np.isclose(frame.Q1, q, atol=1e-12, rtol=0)]
        if args.members:
            group = group.iloc[:args.members]
        for record in group.itertuples():
            initial = np.array([record.initial_delta_m, record.initial_omega, record.initial_delta, record.initial_voltage])
            model = module.VoltageCollapseModel(Q1=float(q))
            with contextlib.redirect_stdout(io.StringIO()):
                _, states = model.simulate(initial, (0, 100), dt=.05, verbose=False,
                                           method='RK45', max_step=.1, rtol=1e-4, atol=1e-6)
            signal = np.full(2000, np.nan)
            signal[:min(len(states), 2000)] = states[:2000, 3]
            np.savez_compressed(args.output / f'condition_{i:03d}_member_{record.member:02d}.npz',
                                signal=signal, initial=initial, parameter=q)
            rows.append(dict(condition_index=i, Q1=q, member=record.member,
                             survived=bool(np.isfinite(signal).all() and np.all(signal > .30))))
        print(f'Power GT {i} complete', flush=True)
    return rows


def kuramoto(args):
    model = rt.load_module('kura_gt_equations', ROOT / 'System_Kuramoto/solver/kuramoto_model.py')
    network = rt.load_module('kura_gt_network', ROOT / 'System_Kuramoto/solver/network_utils.py')
    graph = network.make_ba_network(N=100, m=3, seed=0)
    adjacency, omega = network.get_adjacency_matrix(graph), network.assign_frequencies_es_normalized(graph)
    rows = []
    frozen = pd.read_csv(ROOT / 'data/gt_baselines/kuramoto_forward_runs.csv')
    grid = frozen[(frozen.seed == frozen.seed.min()) & (frozen.horizon == 500)].sort_values('K').K.to_numpy()
    # Include the original lower-K prefix, even when only the paper range is plotted.
    for seed in range(args.members or 20):
        theta0 = None
        # Recreating decimal K by arange adds tiny binary offsets that can diverge
        # in a chaotic continuation. Reuse the actual recorded grid values.
        for k in grid:
            _, theta, order = model.solve_kuramoto_network(adjacency, omega, float(k),
                t_span=(0, 500), dt=.05, theta0=theta0, seed=seed)
            theta0 = theta[-1].copy()
            dwell = float(np.mean(order[-3000:] >= .5))
            rows.append(dict(seed=seed, K=float(k), horizon=500, tail_time=150,
                             tail_mean_r=float(np.mean(order[-3000:])), dwell=dwell,
                             synchronized=dwell >= .8))
        print(f'Kuramoto GT forward member {seed} complete', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', choices=['food', 'power', 'kuramoto'], required=True)
    p.add_argument('--condition', type=int, action='append')
    p.add_argument('--members', type=int)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/gt')
    args = p.parse_args()
    if args.members is not None and not 1 <= args.members <= (50 if args.system == 'power' else 20):
        p.error('members is outside the baseline ensemble range')
    if args.system == 'kuramoto' and args.condition is not None:
        p.error('Continuation cannot skip its lower-K prefix; use --members for a smoke test')
    if args.output.resolve().is_relative_to(ROOT / 'data') or args.output.resolve().is_relative_to(ROOT / 'models'):
        p.error('GT outputs must not overwrite frozen assets')
    args.output = args.output / args.system
    args.output.mkdir(parents=True, exist_ok=True)
    rows = dict(food=food, power=power, kuramoto=kuramoto)[args.system](args)
    pd.DataFrame(rows).to_csv(args.output / 'members.csv', index=False)


if __name__ == '__main__':
    main()
