"""Render the final paper figures from frozen numerical data."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'scdt-mpl'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import runtime as rt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_individual(output):
    table = pd.read_csv(ROOT / 'data/gt_baselines/kuramoto_forward_runs.csv')
    table = table[np.isclose(table.horizon, 500)]
    fig, axis = plt.subplots(figsize=(4.85, 3.35))
    palette = plt.get_cmap('Blues')(np.linspace(.35, .82, table.seed.nunique()))
    for color, (_, group) in zip(palette, table.groupby('seed')):
        group = group.sort_values('K')
        axis.plot(group.K, group.tail_mean_r, color=color, alpha=.58, lw=.78,
                  marker='o', ms=1.7, markeredgewidth=0)
    axis.axhline(.5, color='#b54440', linestyle=(0, (4, 3)), lw=1)
    axis.set(title='Individual forward continuations', xlabel=r'coupling $K$',
             ylabel=r'tail-mean $r$', xlim=(.145, .355), ylim=(-.03, 1.03))
    axis.set_xticks([.15, .20, .25, .30, .35])
    axis.grid(False)
    axis.tick_params(which='both', top=True, right=True, direction='in')
    for spine in axis.spines.values():
        spine.set_linewidth(.8)
        spine.set_color('#202020')
    fig.subplots_adjust(left=.16, right=.98, top=.89, bottom=.18)
    save(fig, output, 'Figure 11')


def save(fig, output, stem):
    for extension in ('png', 'jpg', 'pdf'):
        kwargs = dict(pil_kwargs={'quality': 95}) if extension == 'jpg' else {}
        fig.savefig(output / f'{stem}.{extension}', dpi=320, bbox_inches='tight', pad_inches=.04, **kwargs)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/figures')
    args = p.parse_args()
    if args.output.resolve().is_relative_to(ROOT / 'figures/reference'):
        p.error('Do not overwrite the frozen submission figures')
    args.output.mkdir(parents=True, exist_ok=True)
    notation = rt.load_module('final_notation_renderer', ROOT / 'tools/render_final_notation.py')
    notation.render(ROOT, ROOT, args.output, ROOT / 'data/final_figure_notation_20261001')
    style = rt.load_module('final_trajectory_renderer', ROOT / 'tools/generate_trajectory_figures_with_post.py')
    style.configure_style()
    for system, stem, folder in [('food_chain', 'Figure 3', 'food_chain'),
                                  ('voltage_collapse', 'Figure 6', 'voltage_consistent_final')]:
        cfg = dict(style.SYSTEMS[system])
        traces = pd.read_csv(ROOT / 'data' / folder / 'trajectory.csv')
        if system == 'food_chain':
            reconstruction = pd.read_csv(ROOT / 'data/supplied_vs_reproduced/food_chain_plotted.csv')
        else:
            cfg['post_steps'] = 200
            reconstruction = pd.read_csv(ROOT / 'data/voltage_consistent_final/reconstruction_single_warmup.csv')
            reconstruction = reconstruction[reconstruction.survived]
        fig = style.post_extended_trajectory_figure(traces, reconstruction, cfg)
        save(fig, args.output, stem)
    # Figure 11 has its own original font sizes.
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10.5,
                         'axes.linewidth': .8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
    render_individual(args.output)
    shutil.copy2(ROOT / 'figures/reference/Figure 1.jpg', args.output / 'Figure 1.jpg')
    print(f'Figures written to {args.output}; Figure 1 is a supplied static diagram')


if __name__ == '__main__':
    main()
