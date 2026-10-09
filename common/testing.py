"""Evaluate frozen final settings; never select new points or overwrite references."""
import argparse
import contextlib
import hashlib
import io
import json
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
from threadpoolctl import threadpool_limits


def load_model(args):
    loaded = dict(food=rt.food_load, power=rt.power_load, kuramoto=rt.kuramoto_load)[args.system]()
    if args.readout:
        with np.load(args.readout, allow_pickle=False) as z:
            fitted = z['W_out'].copy(order='K')
        model = loaded[1] if args.system != 'kuramoto' else loaded[0]
        name = 'w_out' if args.system == 'kuramoto' else 'W_out'
        if fitted.shape != getattr(model, name).shape or not np.isfinite(fitted).all():
            raise ValueError('Readout shape/values do not match this frozen reservoir')
        setattr(model, name, fitted)
    return loaded


def run_rate(args):
    output = args.output / args.system / 'rate'
    output.mkdir(parents=True, exist_ok=True)
    if args.system == 'food':
        loaded = load_model(args)
        settings = json.loads((ROOT / 'data/food_rate_single_rule_20261001/evaluation_settings.json').read_text())
    elif args.system == 'power':
        loaded = load_model(args)
        tasks = json.loads((ROOT / 'data/matched_rate_raw_axes_20261001/power/tasks.json').read_text())
        settings = [dict(condition_index=t[1], supplied_stat=t[2], calibration_parameter=t[3],
                         query_parameter=t[4], windows=t[5]) for t in tasks]
    else:
        loaded = load_model(args)
        table = pd.read_csv(ROOT / 'data/final_figure_notation_20261001/scdt_rate.csv')
        settings = [dict(condition_index=i, supplied_stat=r.supplied_stat,
                         calibration_parameter=r.equivalent_parameter) for i, r in table.iterrows()]
    rows = []
    for setting in settings:
        index, sigma = setting['condition_index'], setting['supplied_stat']
        if args.condition is not None and index not in args.condition:
            continue
        member_rows, metadata = [], {}
        with contextlib.redirect_stdout(io.StringIO()):
            if args.system == 'food':
                starts = setting['warmup_starts'][:args.members] if args.members else setting['warmup_starts']
                values = [rt.food_predict(loaded, setting['training_source_index'], start, sigma, 1000) for start in starts]
                member_rows = [dict(member=j, warmup_start=start, **rt.food_status(v))
                               for j, (start, v) in enumerate(zip(starts, values))]
            elif args.system == 'power':
                ends = setting['windows'][:args.members] if args.members else setting['windows']
                values = [rt.power_predict(loaded, setting['query_parameter'], sigma, end, 2000) for end in ends]
                member_rows = [dict(member=j, warmup_end=end, **rt.power_status(loaded[0], v))
                               for j, (end, v) in enumerate(zip(ends, values))]
            else:
                indices = list(range(args.members)) if args.members else None
                values, metadata = rt.kuramoto_predict(loaded, sigma, window_indices=indices)
                dwell, flags = rt.kuramoto_status(values)
                member_rows = [dict(member=j, dwell=float(dwell[j]), synchronized=bool(flags[j]),
                                    finite=bool(np.isfinite(v).all()), tail_mean=float(v[-3000:].mean()))
                               for j, v in enumerate(values)]
        values = np.asarray(values)
        flag = 'synchronized' if args.system == 'kuramoto' else 'survived'
        rows.append(dict(condition_index=index, supplied_stat=sigma,
                         equivalent_parameter=setting['calibration_parameter'],
                         rate=float(np.mean([r[flag] for r in member_rows])), n_members=len(values), **metadata))
        np.savez_compressed(output / f'point_{index:03d}.npz', trajectories=values, supplied_stat=sigma)
        pd.DataFrame(member_rows).to_csv(output / f'point_{index:03d}_members.csv', index=False)
        pd.DataFrame(rows).to_csv(output / 'rate.csv', index=False)
        print(f'{args.system} condition {index}: rate={rows[-1]["rate"]:.2f}', flush=True)
    (output / 'execution.json').write_text(json.dumps(dict(
        system=args.system, task='rate', conditions=len(rows), subset=args.condition,
        members_override=args.members, readout_override=str(args.readout) if args.readout else None,
        independent_conditions=True, target_feeding=False), indent=2))


def run_reconstruction(args):
    if args.system == 'kuramoto':
        raise ValueError('r-only output cannot reproduce oscillator-pair correlations')
    output = args.output / args.system / 'reconstruction'
    output.mkdir(parents=True, exist_ok=True)
    if args.system == 'food':
        loaded = load_model(args)
        settings = pd.read_csv(ROOT / 'data/supplied_vs_reproduced/food_chain_sweep.csv')
    else:
        loaded = load_model(args)
        settings = pd.read_csv(ROOT / 'data/voltage_consistent_final/reconstruction_single_warmup.csv')
    rows = []
    for i, row in settings.iterrows():
        if args.condition is not None and i not in args.condition:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            if args.system == 'food':
                source = int(np.argmin(abs(loaded[2]['ks'] - row.estimated_parameter)))
                signal = rt.food_predict(loaded, source, len(loaded[2]['sequences'][source]) - 500,
                                         row.supplied_std, int(row.n_prediction_steps), row.supplied_secondary_stat)
                status = rt.food_status(signal, tail=50)
                stat = loaded[0].get_stats(signal) if status['survived'] else [np.nan, np.nan]
            else:
                signal = rt.power_predict(loaded, row.Q1, row.supplied_std, int(row.window_end), int(row.horizon))
                status = rt.power_status(loaded[0], signal)
                stat = [status['reproduced_std'], status['reproduced_ac']]
        record = dict(status)
        record.update(condition_index=i, supplied_std=row.supplied_std,
                      reproduced_std=stat[0], reproduced_secondary=stat[1])
        rows.append(record)
        np.savez_compressed(output / f'point_{i:03d}.npz', signal=signal, supplied_std=row.supplied_std)
        print(f'{args.system} reconstruction {i}: survived={status["survived"]}', flush=True)
    pd.DataFrame(rows).to_csv(output / 'measurements_all.csv', index=False)
    pd.DataFrame(rows).query('survived').to_csv(output / 'measurements_noncollapsed.csv', index=False)


def run_selected(args):
    output = args.output / args.system / 'selected'
    output.mkdir(parents=True, exist_ok=True)
    if args.system == 'kuramoto':
        loaded = load_model(args)
        values = json.loads((ROOT / 'data/final_figure_notation_20261001/verified_values.json').read_text())
        for role in ('inter', 'pre', 'post'):
            signal, metadata = rt.kuramoto_predict(loaded, values[role]['S_int'])
            dwell, flags = rt.kuramoto_status(signal)
            np.savez_compressed(output / f'{role}.npz', trajectories=signal, **metadata)
            print(f'{role}: {int(flags.sum())}/20 synchronized', flush=True)
        return
    folder = 'food_chain' if args.system == 'food' else 'voltage_consistent_final'
    points = pd.read_csv(ROOT / 'data' / folder / 'selected_points.csv')
    loaded = load_model(args)
    for row in points[points.role.ne('train')].itertuples():
        with contextlib.redirect_stdout(io.StringIO()):
            if args.system == 'food':
                source = int(np.argmin(abs(loaded[2]['ks'] - row.K)))
                signal = rt.food_predict(loaded, source, len(loaded[2]['sequences'][source]) - 500,
                                         row.stat_1, 2000, row.stat_2, selected_post=row.role == 'post')
                status = rt.food_status(signal, tail=100)
            else:
                end, steps = (26375, 2000) if row.role == 'post' else (9500, 10000)
                signal = rt.power_predict(loaded, row.Q1, row.stat_1, end, steps, row.stat_2)
                status = rt.power_status(loaded[0], signal)
        np.savez_compressed(output / f'{row.role}.npz', signal=signal, supplied_stat=row.stat_1)
        print(f'{row.role}: survived={status["survived"]}', flush=True)


def resolve_readout(system, explicit=None, frozen=False):
    """Never infer a new model from the presence of a training output file."""
    if frozen:
        return None
    if explicit is not None:
        if not explicit.is_file():
            raise ValueError(f'Readout file not found: {explicit}')
        return explicit
    return None


def model_selection(system, readout):
    submitted = dict(food='models/food_chain/checkpoint.npz',
                     power='models/voltage_60000/refitted_readout.npz',
                     kuramoto='models/kuramoto/model_checkpoint.npz')
    path = readout if readout is not None else ROOT / submitted[system]
    return dict(kind='explicit_readout' if readout is not None else 'submitted',
                artifact=str(readout) if readout is not None else submitted[system],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main(system=None):
    p = argparse.ArgumentParser(description=__doc__)
    if system is None:
        p.add_argument('--system', choices=['food', 'power', 'kuramoto'], required=True)
    else:
        p.set_defaults(system=system)
    p.add_argument('--task', choices=['rate', 'reconstruction', 'selected'], default='rate')
    p.add_argument('--condition', type=int, action='append', help='Evaluate only these zero-based settings')
    p.add_argument('--members', type=int, help='Smoke-test subset; omit for the full paper ensemble')
    weights = p.add_mutually_exclusive_group()
    weights.add_argument('--readout', type=Path, help='Explicitly use this refitted readout instead of the submitted model')
    weights.add_argument('--frozen', action='store_true', help='Use the submitted model (already the default; retained for compatibility)')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/evaluation')
    args = p.parse_args()
    try:
        args.readout = resolve_readout(args.system, args.readout, args.frozen)
    except ValueError as error:
        p.error(str(error))
    print(f'Readout: {args.readout if args.readout else "frozen submitted model"}', flush=True)
    if args.members is not None and not 1 <= args.members <= (50 if args.system == 'power' else 20):
        p.error('members is outside the recorded ensemble range')
    if args.output.resolve().is_relative_to(ROOT / 'data') or args.output.resolve().is_relative_to(ROOT / 'models'):
        p.error('Evaluation outputs must not overwrite frozen data/models')
    if args.system == 'kuramoto' and args.task == 'reconstruction':
        p.error('r-only output cannot reproduce oscillator-pair correlations')
    output = args.output / args.system / args.task
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model_selection.json').write_text(json.dumps(model_selection(args.system, args.readout), indent=2) + '\n')
    with threadpool_limits(limits=1):
        {'rate': run_rate, 'reconstruction': run_reconstruction, 'selected': run_selected}[args.task](args)


if __name__ == '__main__':
    main()
