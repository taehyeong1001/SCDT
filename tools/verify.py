"""Check package hashes, coordinates, frozen member decisions and plotted axes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import runtime as rt
import numpy as np
import pandas as pd
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest():
    manifest = json.loads((ROOT / 'provenance/package_manifest.json').read_text())
    for relative, expected in manifest.items():
        if digest(ROOT / relative) != expected:
            raise AssertionError(f'Changed/missing frozen file: {relative}')
    return len(manifest)


def check_rates():
    summaries = {}
    for system in ('food', 'power', 'kuramoto'):
        if system == 'food':
            folder = ROOT / 'data/food_rate_single_rule_20261001'
            table = pd.read_csv(folder / 'scdt_rate.csv')
            gt = pd.read_csv(folder / 'gt_rate.csv')
        elif system == 'power':
            folder = ROOT / 'data/matched_rate_raw_axes_20261001/power'
            table = pd.read_csv(folder / 'scdt_rate.csv')
            gt = pd.read_csv(folder / 'gt_rate.csv')
        else:
            folder = ROOT / 'data/kuramoto_rate_members'
            table = pd.read_csv(ROOT / 'data/final_figure_notation_20261001/scdt_rate.csv')
            gt = pd.read_csv(ROOT / 'data/final_figure_notation_20261001/gt_rate.csv')
        assert len(table) == dict(food=22, power=35, kuramoto=18)[system]
        np.testing.assert_allclose(np.diff(table.supplied_stat), np.diff(table.supplied_stat)[0], atol=1e-12, rtol=0)
        for i, row in table.iterrows():
            name = f'extended_{i:03d}.npz' if system == 'power' else f'point_{i:03d}.npz'
            with np.load(folder / name) as z:
                traces = z['trajectories']
                np.testing.assert_allclose(float(z['supplied_stat']), row.supplied_stat, atol=1e-14, rtol=0)
            assert traces.shape == dict(food=(20, 1000), power=(50, 2000), kuramoto=(20, 10000))[system]
            if system == 'food':
                flags = np.isfinite(traces).all(axis=1) & (traces[:, -500:].mean(axis=1) > .05)
            elif system == 'power':
                flags = np.isfinite(traces).all(axis=1) & (traces > .30).all(axis=1)
            else:
                _, flags = rt.kuramoto_status(traces)
            np.testing.assert_allclose(flags.mean(), row.rate, atol=1e-14, rtol=0)
        summaries[system] = dict(conditions=len(table), members=int(table.n_members.iloc[0]),
            SCDT_crossing=rt.crossing(table.supplied_stat, table.rate, increasing=system == 'kuramoto'),
            GT_crossing=rt.crossing(gt.plotted_stat, gt.rate, increasing=system == 'kuramoto'))
    return summaries


def check_coordinates():
    for folder in ('food_chain', 'voltage_consistent_final'):
        points = pd.read_csv(ROOT / 'data' / folder / 'selected_points.csv')
        train, test = points[points.role.eq('train')], points[points.role.ne('train')]
        np.testing.assert_allclose(test.stat_2, np.polyval(np.polyfit(train.stat_1, train.stat_2, 1), test.stat_1), atol=1e-12, rtol=0)
    values = json.loads((ROOT / 'data/final_figure_notation_20261001/verified_values.json').read_text())
    train = pd.read_csv(ROOT / 'models/kuramoto/training_coordinates.csv')
    for role in ('inter', 'pre', 'post'):
        np.testing.assert_allclose(values[role]['S_int'], np.polyval(np.polyfit(train.stat_1, train.stat_2, 1), values[role]['S_out']), atol=1e-12, rtol=0)
    assert train.stat_1.max() < values['pre']['S_out']
    assert train.stat_2.max() < values['pre']['S_int'] < values['coarse_SCDT_crossing_S_int'] < values['GT_crossing_S_int']
    return True


def check_reconstruction():
    points = pd.read_csv(ROOT / 'data/voltage_consistent_final/reconstruction_single_warmup.csv')
    assert len(points) == 40 and points.survived.all()
    food = pd.read_csv(ROOT / 'data/supplied_vs_reproduced/food_chain_sweep.csv')
    shown = pd.read_csv(ROOT / 'data/supplied_vs_reproduced/food_chain_plotted.csv')
    assert shown.status.eq('survived').all()
    for r in shown.itertuples():
        matches = food[np.isclose(food.supplied_std, r.supplied_std, atol=1e-14, rtol=0)]
        assert len(matches) == 1
        np.testing.assert_allclose(matches.reproduced_std.iloc[0], r.reproduced_std, atol=1e-14, rtol=0)
    visible = shown[shown.supplied_std.between(.120, .134888516590)
                    & shown.reproduced_std.between(.120, .134888516590)]
    return dict(food_all=len(food), food_display_table=len(shown), food_visible=len(visible), power_all=len(points))


def check_new_predictions(output):
    reports = {}
    for system, reference, filename in [
        ('food', ROOT / 'data/food_rate_single_rule_20261001', 'point_{i:03d}.npz'),
        ('power', ROOT / 'data/matched_rate_raw_axes_20261001/power', 'extended_{i:03d}.npz'),
        ('kuramoto', ROOT / 'data/kuramoto_rate_members', 'point_{i:03d}.npz')]:
        folder = output / system / 'rate'
        for path in folder.glob('point_*.npz'):
            i = int(path.stem.split('_')[-1])
            with np.load(path) as z:
                actual = z['trajectories']
            with np.load(reference / filename.format(i=i)) as z:
                expected = z['trajectories'][:len(actual)]
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7, equal_nan=True)
            reports[f'{system}/{i}'] = dict(members=len(actual), steps=actual.shape[1],
                max_absolute_error=float(np.nanmax(abs(actual - expected))))
    for system, reference in [('food', ROOT / 'data/supplied_vs_reproduced/food_chain_sweep.csv'),
                               ('power', ROOT / 'data/voltage_consistent_final/reconstruction_single_warmup.csv')]:
        path = output / system / 'reconstruction/measurements_all.csv'
        if not path.exists():
            continue
        expected = pd.read_csv(reference)
        errors = []
        for row in pd.read_csv(path).itertuples():
            value = float(expected.reproduced_std.iloc[row.condition_index])
            decision = (expected.status.iloc[row.condition_index] == 'survived'
                        if system == 'food' else bool(expected.survived.iloc[row.condition_index]))
            assert bool(row.survived) == decision, f'Reconstruction decision differs: {system}/{row.condition_index}'
            if system == 'power':
                np.testing.assert_allclose(row.reproduced_std, value, atol=1e-7, rtol=1e-7, equal_nan=True)
            if np.isfinite(value) and np.isfinite(row.reproduced_std):
                errors.append(abs(row.reproduced_std - value))
            reports[f'{system}/reconstruction/{row.condition_index}'] = dict(
                absolute_std_error=(float(abs(row.reproduced_std - value))
                                    if np.isfinite(value) and np.isfinite(row.reproduced_std) else None))
        reports[f'{system}/reconstruction_summary'] = dict(
            maximum_std_error=float(max(errors)) if errors else None,
            mean_std_error=float(np.mean(errors)) if errors else None,
            decisions_identical=True,
            exact_statistic_equivalence_required=system == 'power')
    assert reports, f'No supported prediction outputs found in {output}'
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path)
    p.add_argument('--figures', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/verification.json')
    args = p.parse_args()
    result = dict(manifest_files=check_manifest(), coordinate_check=check_coordinates(),
                  rates=check_rates(), reconstruction=check_reconstruction())
    if args.predictions:
        result['fresh_prediction_comparison'] = check_new_predictions(args.predictions)
    if args.figures:
        comparisons = {f'Figure {i}.jpg': digest(args.figures / f'Figure {i}.jpg') == digest(ROOT / f'figures/reference/Figure {i}.jpg') for i in range(1, 12)}
        result['figure_bytes_identical'] = comparisons
        # The submitted Figure 3 was JPEG-reencoded in an external image editor.
        # Verify its raster dimensions and decoded pixels, without copying it over a new render.
        result['figure_visual_comparison'] = {}
        for name, same in comparisons.items():
            if same:
                continue
            actual = np.asarray(Image.open(args.figures / name).convert('RGB'), dtype=float)
            expected = np.asarray(Image.open(ROOT / 'figures/reference' / name).convert('RGB'), dtype=float)
            assert actual.shape == expected.shape, f'Figure dimensions differ: {name}'
            error = abs(actual - expected)
            mae, maximum = float(error.mean()), float(error.max())
            result['figure_visual_comparison'][name] = dict(pixel_MAE=mae, maximum_error=maximum)
            assert name == 'Figure 3.jpg' and mae < 1 and maximum < 40, f'Figure layout/encoding differs: {name}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
