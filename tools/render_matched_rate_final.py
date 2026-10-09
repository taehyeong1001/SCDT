"""Align final Food/Power rates using their displayed raw-panel regressions."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scdt-mpl"))

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


def crossing(x, y):
    matches = np.flatnonzero((y[:-1] >= 50) & (y[1:] < 50))
    if len(matches) != 1:
        raise ValueError("Expected exactly one downward 50% crossing")
    i = int(matches[0])
    fraction = (50 - y[i]) / (y[i + 1] - y[i])
    return float(x[i] + fraction * (x[i + 1] - x[i]))


def render(code, data, output, preview=None, food_data=None, systems=None):
    spec = importlib.util.spec_from_file_location("matched_final_style", code / "tools/generate_realfinal_figures.py")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.configure_style()
    output.mkdir(parents=True, exist_ok=True)
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
    fits = json.loads((data / "raw_panel_fits.json").read_text())
    audit = {}
    for system, stem, folder, param, key in [
        ("food_chain", "Figure 4", "food_chain", "K", "food"),
        ("power_system", "Figure 7", "voltage_consistent_final", "Q1", "power"),
    ]:
        if systems and key not in systems:
            continue
        rate_root = food_data if key == "food" and food_data else data
        rate = pd.read_csv(rate_root / f"{key}_rate.csv")
        selected = pd.read_csv(code / "data" / folder / "selected_points.csv")
        cfg = dict(style.SYSTEMS["food_chain" if key == "food" else "voltage_collapse"])
        fig, _ = style.rate_and_stat_figure(selected, rate, cfg)
        left, right = fig.axes[:2]
        x = rate.representative_stat.to_numpy(float)
        parameter = rate[param].to_numpy(float)
        gt_cross = crossing(parameter, rate.gt_rate.to_numpy(float))
        query = rate.SCDT_original_query_parameter.to_numpy(float) if "SCDT_original_query_parameter" in rate else parameter
        rc_cross = crossing(query, rate.rc_rate.to_numpy(float))
        assert np.all(np.diff(x) > 0)
        fit = fits[system]
        def parameter_to_stat(value):
            return fit["normalized_slope"] * (value - fit["center"]) / fit["scale"] + fit["normalized_intercept"]
        def stat_to_parameter(value):
            return (value - fit["normalized_intercept"]) / fit["normalized_slope"] * fit["scale"] + fit["center"]
        gt_x = parameter_to_stat(parameter)
        np.testing.assert_allclose(stat_to_parameter(gt_x), parameter, atol=1e-12, rtol=0)
        left.lines[0].set_xdata(gt_x)
        left.lines[1].set_xdata(x)
        for child in list(left.child_axes):
            child.remove()
        top = left.secondary_xaxis("top", functions=(stat_to_parameter, parameter_to_stat))
        top.set_xlabel(param, labelpad=4)
        ticks = ([.975, .985, .995, 1.005] if parameter.min() >= .975 - 1e-12 else
                 [.960, .975, .990, 1.005]) if key == "food" else [2.98964, 2.98972, 2.98980, 2.98988, 2.98996]
        top.set_xticks(ticks)
        top.xaxis.set_major_formatter(FormatStrFormatter("%.3f" if key == "food" else "%.5f"))
        top.tick_params(labelsize=8, length=3.5, width=.8, pad=2, direction="in")
        for line in list(left.lines):
            if line.get_label() == "critical point":
                line.remove()
        marked_sigma = crossing(x, rate.rc_rate.to_numpy(float))
        marked_parameter = float(stat_to_parameter(marked_sigma))
        marked_rate = float(np.interp(marked_sigma, x, rate.rc_rate))
        np.testing.assert_allclose(marked_rate, 50.0, atol=1e-9)
        left.axvline(marked_sigma, color=style.RED, linestyle=(0, (4, 3)), lw=1.15, label="critical point")
        extent = np.r_[gt_x, x]
        padding = .045 * float(np.ptp(extent))
        left.set_xlim(float(extent.min() - padding), float(extent.max() + padding))
        left.legend(loc="best", handlelength=2.2)

        points = pd.read_csv(data / f"raw_panel_{key}.csv")
        assert len(points) == (30 if key == "food" else 40)
        xx, yy = points.parameter.to_numpy(float), points.raw_std.to_numpy(float)
        padding = .055 * float(np.ptp(xx))
        xlim = (float(xx.min() - padding), float(xx.max() + padding))
        line_x = np.linspace(*xlim, 200)
        line_y = fit["normalized_slope"] * (line_x - fit["center"]) / fit["scale"] + fit["normalized_intercept"]
        extent = np.r_[yy, line_y]
        padding_y = .075 * float(np.ptp(extent))
        right.clear()
        right.plot(line_x, line_y, color=style.RED, lw=1, linestyle=(0, (3, 2.5)))
        right.scatter(xx, yy, s=22, color=style.POINT_BLUE, edgecolor="white", linewidth=.4, zorder=3)
        right.set(xlabel=param, ylabel=cfg["representative"], xlim=xlim,
                  ylim=(float(extent.min() - padding_y), float(extent.max() + padding_y)))
        style.finish_axis(right)
        style.compact_parameter_axis(right, param, bins=4)
        right.yaxis.set_major_locator(style.MaxNLocator(5))
        style.add_panel_label(right, "(b)", x=-.22)
        right.grid(False)
        np.testing.assert_allclose(left.lines[0].get_ydata(), rate.gt_rate, atol=1e-12)
        np.testing.assert_allclose(left.lines[1].get_ydata(), rate.rc_rate, atol=1e-12)
        np.testing.assert_array_equal(left.lines[1].get_xdata(), x)
        fig.savefig(output / f"{stem}.jpg", dpi=320, bbox_inches="tight", pad_inches=.04,
                    pil_kwargs={"quality": 95})
        if preview:
            fig.savefig(preview / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=.04)
        plt.close(fig)
        audit[key] = dict(GT_crossing_actual_parameter=gt_cross,
                          GT_crossing_mapped_stat=float(parameter_to_stat(gt_cross)),
                          SCDT_crossing_original_query_parameter=rc_cross,
                          SCDT_crossing_equivalent_parameter=marked_parameter,
                          marked_parameter=marked_parameter, marked_sigma=marked_sigma,
                          marked_SCDT_rate=marked_rate, marker_rule="SCDT 50% crossing",
                          raw_panel_count=len(points), raw_panel_refitted=False,
                          upper_axis="Inverse of the displayed raw-panel regression",
                          axis_rule="GT x=f_b(actual parameter); SCDT x=actual supplied statistic; top axis=f_b inverse")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    code_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--code-root", type=Path, default=code_default)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--food-data-root", type=Path)
    parser.add_argument("--system", choices=("food", "power"), action="append")
    args = parser.parse_args()
    updated = args.code_root / "data/matched_rate_raw_axes_20261001"
    data = args.data_root or (updated if updated.is_dir() else args.code_root / "data/matched_rate_final_20260930")
    output = args.output or args.code_root.parent / "script_0929"
    food_data = args.food_data_root
    if food_data is None and args.data_root is None:
        single_rule = args.code_root / "data/food_rate_single_rule_20261001"
        if single_rule.is_dir():
            food_data = single_rule
        elif data != updated:
            candidate = args.code_root / "data/food_rate_final_h1000_tail500_20260930"
            if candidate.is_dir():
                food_data = candidate
    result = render(args.code_root, data, output, args.preview, food_data, args.system)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
