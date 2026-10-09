"""Render notation-only updates from frozen final figure data, without simulation."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scdt-mpl"))

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(fig, output, stem):
    for extension in ("jpg", "png"):
        options = {"pil_kwargs": {"quality": 95}} if extension == "jpg" else {}
        fig.savefig(output / f"{stem}.{extension}", dpi=320,
                    bbox_inches="tight", pad_inches=.04, **options)
    plt.close(fig)


def render(code, style_code, output, notation_data):
    output.mkdir(parents=True, exist_ok=True)
    style = load_module("notation_style", style_code / "tools/generate_realfinal_figures.py")
    style.configure_style()
    for system, folder, stem in [("food_chain", "food_chain", "Figure 2"),
                                 ("voltage_collapse", "voltage_consistent_final", "Figure 5")]:
        points = pd.read_csv(code / "data" / folder / "selected_points.csv")
        save(style.stat_space_figure(points, style.SYSTEMS[system]), output, stem)

    rates = load_module("notation_matched_rates", style_code / "tools/render_matched_rate_final.py")
    rate_audit = rates.render(style_code, code / "data/matched_rate_raw_axes_20261001",
                              output, output, code / "data/food_rate_single_rule_20261001")

    values = json.loads((notation_data / "verified_values.json").read_text())
    train = pd.DataFrame(values["training"])
    pre = values["pre"]
    a, b = (values["stat_space_fit"][key] for key in ("slope", "intercept"))
    fig, axis = plt.subplots(figsize=(3.5, 3.2))
    tests = [("inter", values["inter"]["S_out"], values["inter"]["S_int"], "#249575"),
             ("pre", pre["S_out"], pre["S_int"], "#d59a27"),
             ("post", values["post"]["S_out"], values["post"]["S_int"], style.RED)]
    x = np.r_[train.stat_1.to_numpy(), [v[1] for v in tests]]
    line = np.linspace(x.min()-.06*np.ptp(x), x.max()+.06*np.ptp(x), 200)
    axis.plot(line, a*line+b, color="#a6a6a6", ls=(0, (3, 3)), lw=1, zorder=1)
    axis.scatter(train.stat_1, train.stat_2, s=14, color=style.POINT_BLUE, zorder=3)
    for label, out, sigma, color in tests:
        axis.scatter(out, sigma, s=33, color=color, edgecolor="#252525", linewidth=.7, zorder=4)
        axis.annotate(label, (out, sigma), xytext=(-5, 6), textcoords="offset points",
                      ha="right", va="bottom", fontsize=8)
    axis.set(xlabel=style.SYSTEMS["kuramoto"]["stat_x"],
             ylabel=style.SYSTEMS["kuramoto"]["stat_y"], title="Explosive Synchronization")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    style.finish_axis(axis)
    fig.tight_layout()
    save(fig, output, "Figure 8")

    with np.load(notation_data / "selected_trajectories.npz") as traces:
        signals = [(traces["inter_gt"], traces["inter_scdt"]),
                   (traces["pre_gt"], traces["pre_scdt"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.55, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.115, top=.85, hspace=.45, wspace=.10)
    time = np.arange(1, 2001)*.05
    for row, pair in enumerate(signals):
        for col, signal in enumerate(pair):
            axis = axes[row, col]
            axis.plot(time, signal[:2000], color=style.BLUE if col == 0 else style.RED, lw=1)
            axis.axhline(.5, color="#929292", ls=(0, (3, 3)), lw=.8)
            axis.set(title="GT" if col == 0 else "SCDT", xlim=(0, 100), ylim=(-.02, 1.02))
            axis.set_xticks(np.linspace(0, 100, 6))
            axis.set_yticks([0, .25, .5, .75, 1])
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if col == 0:
                axis.set_ylabel("$r(t)$")
            if row == 1:
                axis.set_xlabel("Time")
            style.finish_axis(axis)
    fig.suptitle("Explosive Synchronization", fontsize=12, y=.985)
    fig.text(.01, .91, f"(a)  Interpolation,  K={values['inter']['actual_target_K']:.9f}", fontsize=10)
    fig.text(.01, .467, f"(b)  Pre-sync extrapolation,  K={pre['actual_target_K']:.9f}", fontsize=10)
    save(fig, output, "Figure 9")

    sc = pd.read_csv(notation_data / "scdt_rate.csv")
    gt = pd.read_csv(notation_data / "gt_rate.csv")
    raw = pd.read_csv(notation_data / "raw_panel_kuramoto.csv")
    slope, intercept = (values["axis_calibration"][key] for key in ("slope", "intercept"))
    fig = plt.figure(figsize=(7.55, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.4, 1], wspace=.43)
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    left.plot(gt.plotted_stat, gt.rate, "o-", color="#202020", ms=3.4, lw=1.15, label="GT")
    left.plot(sc.supplied_stat, sc.rate, "s-", color=style.BLUE, ms=3.3, lw=1.15, label="SCDT")
    left.axvline(values["coarse_SCDT_crossing_S_int"], color=style.RED,
                 ls=(0, (4, 3)), lw=1, label="critical point")
    left.set(xlabel=style.SYSTEMS["kuramoto"]["representative"], ylabel="synchronization rate",
             ylim=(-.04, 1.04), xlim=(slope*.19+intercept-.005, slope*.30+intercept+.005))
    left.xaxis.set_major_locator(MaxNLocator(5))
    left.yaxis.set_major_locator(MaxNLocator(5))
    top = left.secondary_xaxis("top", functions=(lambda s: (s-intercept)/slope, lambda k: slope*k+intercept))
    top.set_xlabel("K")
    top.set_xticks([.2, .225, .25, .275, .3])
    top.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    left.legend(loc="upper left", fontsize=8, frameon=False)
    k_column = "K" if "K" in raw else "parameter"
    stat_column = next(c for c in ("S_int", "S_int_mean", "statistic", "raw_stat", "stat_2", "mean_S_int") if c in raw)
    right.scatter(raw[k_column], raw[stat_column], color=style.BLUE, s=12, edgecolor="white", lw=.35, zorder=3)
    line = np.linspace(raw[k_column].min(), raw[k_column].max(), 100)
    right.plot(line, slope*line+intercept, color=style.RED, ls=(0, (3, 3)), lw=1)
    right.set(xlabel="K", ylabel=style.SYSTEMS["kuramoto"]["representative"])
    right.xaxis.set_major_locator(MaxNLocator(3))
    right.yaxis.set_major_locator(MaxNLocator(4))
    for axis in (left, right):
        style.finish_axis(axis)
    fig.suptitle("Explosive Synchronization", fontsize=12, y=.99)
    fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.75)
    fig.text(.015, .82, "(a)", fontsize=10)
    fig.text(.725, .82, "(b)", fontsize=10)
    np.testing.assert_array_equal(left.lines[0].get_ydata(), gt.rate)
    np.testing.assert_array_equal(left.lines[1].get_ydata(), sc.rate)
    save(fig, output, "Figure 10")
    (output / "notation_audit.json").write_text(json.dumps({
        "rate_units_unchanged": True, "simulations_rerun": False,
        "food_power_crossings": rate_audit,
        "kuramoto_crossing": values["coarse_SCDT_crossing_S_int"],
        "inter_title_K": f"{values['inter']['actual_target_K']:.9f}",
        "pre_title_K": f"{pre['actual_target_K']:.9f}",
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--style-code-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notation-data", type=Path)
    args = parser.parse_args()
    render(args.code_root, args.style_code_root or args.code_root,
           args.output or args.code_root / "outputs/figures",
           args.notation_data or args.code_root / "data/final_figure_notation_20261001")


if __name__ == "__main__":
    main()
