#!/usr/bin/env python3
"""Create the paper-style SCDT figures from the final selected CSV files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


BLUE = "#2c5d8a"
POINT_BLUE = "#315f8c"
RED = "#b54440"
GREEN = "#2b8c6b"
ORANGE = "#d18b1f"
POST_RED = "#b83b3b"
REFERENCE = "#8b8b8b"

SYSTEMS = {
    "food_chain": {
        "label": "Tri-Trophic Food Chain Model",
        "parameter": "K",
        "stat_x": r"$\sigma$",
        "stat_y": r"$H_{\mathrm{perm}}$",
        "representative": r"$\sigma$",
        "trajectory_y": "P",
        "critical": 1.0,
        "rate_label": "survival rate (%)",
        "rate_scale": 1.0,
        "prefix": "F",
        "reconstruction_limits": (0.120, 0.134888516590),
    },
    "voltage_collapse": {
        "label": "Power System Model",
        "parameter": "Q1",
        "stat_x": r"$\sigma$",
        "stat_y": r"$\rho_1$",
        "representative": r"$\sigma$",
        "trajectory_y": "V",
        "critical": 2.9898256,
        "rate_label": "survival rate (%)",
        "rate_scale": 1.0,
        "prefix": "V",
        "reconstruction_limits": (0.031447328630, 0.032995647214),
    },
    "kuramoto": {
        "label": "Explosive Synchronization",
        "parameter": "K",
        "stat_x": r"$S_{\mathrm{out}}$",
        "stat_y": r"$S_{\mathrm{int}}$",
        "representative": r"$S_{\mathrm{int}}$",
        "trajectory_y": "r(t)",
        "critical": 0.2371,
        "rate_label": "synchronization rate",
        "rate_scale": 1.0,
        "prefix": "K",
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#202020",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.15,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, png_path: Path, pdf: PdfPages) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.04)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left")


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#202020")


def square_reconstruction_limits(
    frame: pd.DataFrame, padding_fraction: float = 0.035
) -> tuple[float, float]:
    lower = float(min(frame["supplied_std"].min(), frame["reproduced_std"].min()))
    upper = float(max(frame["supplied_std"].max(), frame["reproduced_std"].max()))
    padding = max((upper - lower) * padding_fraction, np.finfo(float).eps)
    return lower - padding, upper + padding


def stat_space_figure(points: pd.DataFrame, cfg: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.75, 3.30))
    train = points[points["role"] == "train"]
    fit = np.polyfit(train["stat_1"], train["stat_2"], 1)
    x_pad = max((points["stat_1"].max() - points["stat_1"].min()) * 0.06, 1e-8)
    x_line = np.linspace(points["stat_1"].min() - x_pad, points["stat_1"].max() + x_pad, 200)
    ax.plot(x_line, np.polyval(fit, x_line), color=REFERENCE, lw=1.0,
            linestyle=(0, (3, 2.5)), alpha=0.72, zorder=1)
    train_marker_size = 18 if cfg["prefix"] == "F" else 26
    test_marker_size = 52 if cfg["prefix"] == "F" else 58
    ax.scatter(train["stat_1"], train["stat_2"], s=train_marker_size, color=POINT_BLUE,
               edgecolor="white", linewidth=0.7, zorder=3)

    point_style = {
        "inter": (GREEN, "inter"),
        "pre": (ORANGE, "pre"),
        "post": (POST_RED, "post"),
    }
    for role, (color, label) in point_style.items():
        row = points[points["role"] == role].iloc[0]
        ax.scatter(row["stat_1"], row["stat_2"], s=test_marker_size, color=color,
                   edgecolor="#222222", linewidth=0.9, zorder=4)
        ax.annotate(
            label,
            (row["stat_1"], row["stat_2"]),
            xytext=(-7, 7),
            textcoords="offset points",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    ax.set_title(cfg["label"], fontweight="normal", pad=6)
    ax.set_xlabel(cfg["stat_x"])
    ax.set_ylabel(cfg["stat_y"])
    finish_axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.tight_layout(pad=0.7)
    return fig


def reconstruction_axis(ax: plt.Axes, frame: pd.DataFrame, cfg: dict) -> None:
    clip_lo, clip_hi = cfg["reconstruction_limits"]
    frame = frame[
        frame["supplied_std"].between(clip_lo, clip_hi)
        & frame["reproduced_std"].between(clip_lo, clip_hi)
    ]
    lo, hi = square_reconstruction_limits(frame)
    ax.plot([lo, hi], [lo, hi], color=REFERENCE, lw=0.95,
            linestyle=(0, (3, 2.5)), label=r"$y=x$", zorder=1)
    ax.scatter(frame["supplied_std"], frame["reproduced_std"], s=17,
               facecolor="#7fa9c9", edgecolor="white", linewidth=0.35,
               alpha=0.82, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Supplied $\sigma$")
    ax.set_ylabel(r"Reproduced $\sigma$")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(loc="upper left", handlelength=2.0)
    finish_axis(ax)


def overlay_trajectory_figure(
    frame: pd.DataFrame, cfg: dict, reconstruction: pd.DataFrame
) -> plt.Figure:
    fig = plt.figure(figsize=(7.35, 3.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.72], wspace=0.34, hspace=0.28)
    ax_reconstruction = fig.add_subplot(gs[:, 0])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    reconstruction_axis(ax_reconstruction, reconstruction, cfg)
    add_panel_label(ax_reconstruction, "(a)", x=-0.22, y=1.02)

    for ax, role, panel in zip(axes, ["inter", "pre"], ["(b)", "(c)"]):
        part = frame[frame["role"] == role].reset_index(drop=True)
        if cfg["prefix"] == "V":
            part = part.iloc[:1000].reset_index(drop=True)
        x = np.arange(len(part))
        if cfg["prefix"] == "F":
            gt, rc = part["gt_P"], part["rc_P"]
        else:
            gt, rc = part["gt"], part["rc"]
        ax.plot(x, gt, color=BLUE, lw=1.0)
        ax.plot(x, rc, color=RED, lw=0.95, linestyle=(0, (4, 2.2)))
        ax.set_xlim(0, max(len(part) - 1, 1))
        ax.set_ylabel(cfg["trajectory_y"], labelpad=5)
        ax.tick_params(axis="y", labelleft=False)
        add_panel_label(ax, panel, x=-0.15, y=0.46)
        finish_axis(ax)
    axes[0].tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time step")
    axes[-1].xaxis.set_major_locator(MaxNLocator(5, integer=True))
    fig.suptitle(cfg["label"], fontsize=11, fontweight="normal", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.15)
    return fig


def kuramoto_trajectory_figure(frame: pd.DataFrame, points: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25), sharex=True, sharey=True)
    for row_index, (role, panel, condition) in enumerate(
        [("inter", "(a)", "Interpolation"), ("pre", "(b)", "Pre-sync extrapolation")]
    ):
        part = frame[frame["role"] == role].reset_index(drop=True)
        parameter = points[points["role"] == role]["K"].iloc[0]
        for col_index, (series, title, color) in enumerate(
            [("gt", "GT", BLUE), ("rc", "SCDT", RED)]
        ):
            ax = axes[row_index, col_index]
            ax.plot(part["time"], part[series], color=color, lw=1.0, linestyle="-")
            ax.axhline(0.5, color=REFERENCE, lw=0.8, linestyle=(0, (3, 2.5)), zorder=0)
            ax.set_title(title, fontsize=9.2, pad=3, fontweight="normal")
            ax.set_xlim(part["time"].min(), part["time"].max())
            ax.set_ylim(-0.02, 1.02)
            finish_axis(ax)
        axes[row_index, 0].set_ylabel("r(t)")
        parameter_label = f"{parameter:.9f}"
        axes[row_index, 0].text(
            -0.13, 1.13, f"{panel}  {condition},  K={parameter_label}",
            transform=axes[row_index, 0].transAxes, ha="left", va="bottom", fontsize=9.3
        )
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.suptitle("Explosive Synchronization", fontsize=11, fontweight="normal", y=0.995)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.12,
                        wspace=0.10, hspace=0.43)
    return fig


def representative_mapping(points: pd.DataFrame, rate: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    parameter = cfg["parameter"]
    train = points[points["role"] == "train"]
    stat_column = "stat_2" if cfg["prefix"] == "K" else "stat_1"
    fit = np.polyfit(train[parameter], train[stat_column], 1)
    source_x = rate[parameter].to_numpy(dtype=float)
    if "representative_stat" in rate.columns:
        values = rate["representative_stat"].to_numpy(dtype=float)
    else:
        values = np.polyval(fit, source_x)
    return pd.DataFrame({parameter: source_x, "representative_stat": values})


def compact_parameter_axis(ax: plt.Axes, parameter: str, bins: int = 5) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(bins))
    if parameter == "Q1":
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))


def add_bifurcation_axis(
    ax: plt.Axes, slope: float, intercept: float, parameter: str
) -> plt.Axes:
    def stat_to_parameter(value):
        return (np.asarray(value) - intercept) / slope

    def parameter_to_stat(value):
        return slope * np.asarray(value) + intercept

    top = ax.secondary_xaxis(
        "top", functions=(stat_to_parameter, parameter_to_stat)
    )
    top.set_xlabel(parameter, labelpad=4)
    top.xaxis.set_major_locator(MaxNLocator(5))
    if parameter == "Q1":
        top.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    else:
        top.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    top.xaxis.get_offset_text().set_visible(False)
    top.tick_params(labelsize=8, length=3.5, width=0.8, pad=2, direction="in")
    return top


def rate_and_stat_figure(
    points: pd.DataFrame, rate: pd.DataFrame, cfg: dict
) -> tuple[plt.Figure, pd.DataFrame]:
    parameter = cfg["parameter"]
    mapping = representative_mapping(points, rate, cfg)
    fig = plt.figure(figsize=(7.55, 3.20))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.30)
    ax_rate = fig.add_subplot(gs[0, 0])
    ax_stat = fig.add_subplot(gs[0, 1])

    map_x = mapping[parameter].to_numpy(dtype=float)
    map_y = mapping["representative_stat"].to_numpy(dtype=float)
    fit = np.polyfit(map_x, map_y, 1)
    slope, intercept = fit
    x = map_y
    gt = rate["gt_rate"].to_numpy(dtype=float) * cfg["rate_scale"]
    rc = rate["rc_rate"].to_numpy(dtype=float) * cfg["rate_scale"]
    ax_rate.plot(x, gt, color="#202020", marker="o", markersize=2.7,
                 lw=1.05, markevery=max(1, len(x) // 12), label="GT")
    ax_rate.plot(x, rc, color=BLUE, marker="s", markersize=2.6,
                 lw=1.05, markevery=max(1, len(x) // 12), label="SCDT")
    critical_parameter = cfg["critical"]
    if cfg["prefix"] == "K":
        crossing_indices = np.flatnonzero((rc[:-1] < 0.5) & (rc[1:] >= 0.5))
        if len(crossing_indices) == 0:
            raise ValueError("Kuramoto rate has no 50% synchronization crossing")
        index = int(crossing_indices[0])
        fraction = (0.5 - rc[index]) / (rc[index + 1] - rc[index])
        critical_parameter = float(
            map_x[index] + fraction * (map_x[index + 1] - map_x[index])
        )
    critical_label = "critical point"
    ax_rate.axvline(np.polyval(fit, critical_parameter), color=RED,
                    linestyle=(0, (4, 3)), lw=1.15, label=critical_label)
    ax_rate.set_xlabel(cfg["representative"])
    ax_rate.set_ylabel(cfg["rate_label"])
    if cfg["prefix"] == "K":
        ax_rate.set_ylim(-0.04, 1.04)
    else:
        ax_rate.set_ylim(-3, 103)
    finish_axis(ax_rate)
    ax_rate.legend(loc="best", handlelength=2.2)
    stat_span = max(float(np.ptp(map_y)), np.finfo(float).eps)
    decimals = max(2, min(5, int(np.ceil(-np.log10(stat_span / 4.0)))))
    ax_rate.xaxis.set_major_locator(MaxNLocator(5))
    ax_rate.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    ax_rate.xaxis.get_offset_text().set_visible(False)
    add_bifurcation_axis(ax_rate, slope, intercept, parameter)
    add_panel_label(ax_rate, "(a)", y=1.19)

    sample_count = min(12, len(mapping))
    sample_idx = np.unique(np.linspace(0, len(mapping) - 1, sample_count, dtype=int))
    line_x = np.linspace(map_x.min(), map_x.max(), 200)
    ax_stat.scatter(map_x[sample_idx], map_y[sample_idx], s=22, color=POINT_BLUE,
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax_stat.plot(line_x, np.polyval(fit, line_x), color=RED,
                 linestyle=(0, (3, 2.5)), lw=1.0, zorder=2)
    ax_stat.set_xlabel(parameter)
    ax_stat.set_ylabel(cfg["representative"])
    finish_axis(ax_stat)
    compact_parameter_axis(ax_stat, parameter, bins=3)
    if parameter == "Q1":
        ax_stat.tick_params(axis="x", labelsize=8)
    ax_stat.yaxis.set_major_locator(MaxNLocator(4))
    add_panel_label(ax_stat, "(b)", x=-0.22)

    fig.suptitle(cfg["label"], fontsize=11, fontweight="normal", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.75, bottom=0.20)
    return fig, mapping


def copy_source_data(source: Path, output: Path) -> None:
    data_out = output / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    for system in SYSTEMS:
        if system == "voltage_collapse" and (Path(__file__).resolve().parent / "voltage_active.json").exists():
            continue
        system_out = data_out / system
        system_out.mkdir(parents=True, exist_ok=True)
        for name in ["selected_points.csv", "trajectory.csv", "trajectory_summary.csv", "rate.csv", "metrics.json"]:
            path = source / system / "data" / name
            if path.exists():
                shutil.copy2(path, system_out / name)


def load_reconstruction_data(script_dir: Path, system: str) -> pd.DataFrame:
    if system == "food_chain":
        return pd.read_csv(
            script_dir / "data/supplied_vs_reproduced/food_chain_plotted.csv"
        )[["supplied_std", "reproduced_std"]]
    if system == "voltage_collapse":
        frame = pd.read_csv(
            script_dir / "data/voltage_active/reconstruction_single_warmup.csv"
        )
        selected = frame[frame["survived"]][["supplied_std", "reproduced_std"]]
        lo, hi = SYSTEMS[system]["reconstruction_limits"]
        return selected[
            selected["supplied_std"].between(lo, hi)
            & selected["reproduced_std"].between(lo, hi)
        ]
    raise ValueError(f"No scalar reconstruction panel is defined for {system}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Optional historical final_selected directory; default uses local data."
    )
    parser.add_argument("--output", type=Path, default=script_dir.parent / "figure")
    args = parser.parse_args()
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source is not None:
        copy_source_data(args.source, script_dir)

    manifest = []
    with PdfPages(args.output / "realfinal_figures.pdf") as pdf:
        for system, cfg in SYSTEMS.items():
            source = (args.source / system / "data" if args.source is not None
                      else script_dir / "data" / system)
            active_voltage = system == "voltage_collapse" and (script_dir / "voltage_active.json").exists()
            if active_voltage:
                source = script_dir / "data/voltage_active"
            points = pd.read_csv(source / "selected_points.csv")
            if active_voltage:
                candidate_path = source / "post_200_candidate_trajectory.csv"
                if candidate_path.exists():
                    candidate = pd.read_csv(candidate_path).iloc[0]
                    post_mask = points["role"] == "post"
                    points.loc[
                        post_mask,
                        ["Q1", "stat_1", "stat_2", "coordinate_source"],
                    ] = [
                        float(candidate["Q1"]),
                        float(candidate["supplied_std"]),
                        float(candidate["supplied_ac"]),
                        "final near-critical displayed post query",
                    ]
            trajectory = pd.read_csv(source / "trajectory.csv")
            extended_rate = script_dir / "data" / system / "rate_extended.csv"
            rate = pd.read_csv(extended_rate if extended_rate.exists() and not active_voltage else source / "rate.csv")
            figure_dir = args.output / system
            figure_dir.mkdir(parents=True, exist_ok=True)

            stat_path = figure_dir / f"{cfg['prefix']}1_stat_space.png"
            save_figure(stat_space_figure(points, cfg), stat_path, pdf)
            manifest.append((system, "stat-space", stat_path.relative_to(args.output)))

            trajectory_path = figure_dir / f"{cfg['prefix']}2_trajectory.png"
            if system == "kuramoto":
                trajectory_fig = kuramoto_trajectory_figure(trajectory, points)
            else:
                reconstruction = load_reconstruction_data(script_dir, system)
                trajectory_fig = overlay_trajectory_figure(trajectory, cfg, reconstruction)
            save_figure(trajectory_fig, trajectory_path, pdf)
            manifest.append((system, "trajectory", trajectory_path.relative_to(args.output)))

            rate_path = figure_dir / f"{cfg['prefix']}3_rate_and_stat.png"
            rate_fig, mapping = rate_and_stat_figure(points, rate, cfg)
            save_figure(rate_fig, rate_path, pdf)
            mapping_dir = script_dir / "data" / ("voltage_active" if active_voltage else system)
            mapping_dir.mkdir(parents=True, exist_ok=True)
            mapping.to_csv(mapping_dir / "representative_stat_mapping.csv", index=False)
            manifest.append((system, "rate-and-stat", rate_path.relative_to(args.output)))

    pd.DataFrame(manifest, columns=["system", "figure_type", "path"]).to_csv(
        script_dir / "figure_manifest.csv", index=False
    )


if __name__ == "__main__":
    main()
