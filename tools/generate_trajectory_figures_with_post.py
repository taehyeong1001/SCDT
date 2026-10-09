#!/usr/bin/env python3
"""Generate the revised Food, Power-system, and Kuramoto trajectory panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


BLUE = "#2c5d8a"
RED = "#b54440"
REFERENCE = "#8b8b8b"

SYSTEMS = {
    "food_chain": {
        "label": "Tri-Trophic Food Chain Model",
        "trajectory_y": "P",
        "prefix": "F",
        "data_dir": "food_chain",
        "gt_column": "gt_P",
        "rc_column": "rc_P",
        "reconstruction_limits": (0.120, 0.134888516590),
        "post_steps": 2000,
    },
    "voltage_collapse": {
        "label": "Power System Model",
        "trajectory_y": "V",
        "prefix": "V",
        "data_dir": "voltage_active",
        "gt_column": "gt",
        "rc_column": "rc",
        "reconstruction_limits": (0.031447328630, 0.032995647214),
        "post_steps": 1000,
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


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#202020")


def add_panel_label(
    ax: plt.Axes, label: str, *, x: float = -0.13, y: float = 1.04
) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left")


def square_reconstruction_limits(frame: pd.DataFrame) -> tuple[float, float]:
    lower = float(min(frame["supplied_std"].min(), frame["reproduced_std"].min()))
    upper = float(max(frame["supplied_std"].max(), frame["reproduced_std"].max()))
    padding = max((upper - lower) * 0.035, np.finfo(float).eps)
    return lower - padding, upper + padding


def reconstruction_axis(ax: plt.Axes, frame: pd.DataFrame, cfg: dict) -> None:
    clip_lo, clip_hi = cfg["reconstruction_limits"]
    frame = frame[
        frame["supplied_std"].between(clip_lo, clip_hi)
        & frame["reproduced_std"].between(clip_lo, clip_hi)
    ]
    lo, hi = square_reconstruction_limits(frame)
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=REFERENCE,
        lw=0.95,
        linestyle=(0, (3, 2.5)),
        label=r"$y=x$",
        zorder=1,
    )
    ax.scatter(
        frame["supplied_std"],
        frame["reproduced_std"],
        s=17,
        facecolor="#7fa9c9",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=2,
    )
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


def load_reconstruction_data(code_dir: Path, system: str) -> pd.DataFrame:
    if system == "food_chain":
        return pd.read_csv(
            code_dir / "data/supplied_vs_reproduced/food_chain_plotted.csv"
        )[["supplied_std", "reproduced_std"]]

    frame = pd.read_csv(
        code_dir / "data/voltage_active/reconstruction_single_warmup.csv"
    )
    return frame.loc[frame["survived"], ["supplied_std", "reproduced_std"]]


def post_extended_trajectory_figure(
    frame: pd.DataFrame, reconstruction: pd.DataFrame, cfg: dict
) -> plt.Figure:
    fig = plt.figure(figsize=(7.35, 5.62))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.02, 1.72],
        height_ratios=[1.0, 1.0, 1.05],
        wspace=0.34,
        hspace=0.53,
    )

    ax_reconstruction = fig.add_subplot(gs[:2, 0])
    ax_inter = fig.add_subplot(gs[0, 1])
    ax_pre = fig.add_subplot(gs[1, 1])
    nested = gs[2, :].subgridspec(1, 2, wspace=0.18)
    ax_post_gt = fig.add_subplot(nested[0, 0])
    ax_post_rc = fig.add_subplot(nested[0, 1], sharey=ax_post_gt)

    reconstruction_axis(ax_reconstruction, reconstruction, cfg)
    add_panel_label(ax_reconstruction, "(a)", x=-0.22, y=1.02)

    conditions = {
        "inter": "interpolation",
        "pre": "pre-collapse extrapolation",
    }
    for ax, role, panel in ((ax_inter, "inter", "(b)"), (ax_pre, "pre", "(c)")):
        part = frame.loc[frame["role"] == role].reset_index(drop=True)
        if cfg["prefix"] == "V":
            part = part.iloc[:1000].reset_index(drop=True)
        x = np.arange(len(part))
        ax.plot(x, part[cfg["gt_column"]], color=BLUE, lw=1.0)
        ax.plot(
            x,
            part[cfg["rc_column"]],
            color=RED,
            lw=0.95,
            linestyle=(0, (4, 2.2)),
        )
        ax.set_xlim(0, max(len(part) - 1, 1))
        ax.set_ylabel(cfg["trajectory_y"], labelpad=5)
        ax.tick_params(axis="y", labelleft=False)
        ax.set_title(
            f"{panel}  {conditions[role]}",
            loc="left",
            x=-0.15,
            pad=8,
            fontsize=10,
            fontweight="normal",
        )
        finish_axis(ax)
    ax_inter.tick_params(labelbottom=False)
    ax_pre.set_xlabel("Time step")
    ax_pre.xaxis.set_major_locator(MaxNLocator(5, integer=True))

    post = frame.loc[frame["role"] == "post"].reset_index(drop=True)
    post = post.iloc[: cfg["post_steps"]].reset_index(drop=True)
    post_x = np.arange(len(post))
    post_series = (
        (ax_post_gt, cfg["gt_column"], "GT", BLUE),
        (ax_post_rc, cfg["rc_column"], "SCDT", RED),
    )
    finite_values = np.concatenate(
        [post[cfg["gt_column"]].dropna(), post[cfg["rc_column"]].dropna()]
    )
    # Negative voltage after numerical collapse is outside the physical plotting range.
    lower = 0.0 if cfg["prefix"] == "V" else min(0.0, float(np.nanmin(finite_values)))
    upper = float(np.nanmax(finite_values))
    margin = max((upper - lower) * 0.04, 0.01)

    for ax, column, title, color in post_series:
        ax.plot(post_x, post[column], color=color, lw=1.0)
        ax.set_xlim(0, max(len(post) - 1, 1))
        if cfg["prefix"] == "V":
            ax.set_ylim(0.70, 0.90)
        else:
            ax.set_ylim(lower - margin, upper + margin)
        ax.set_title(title, fontsize=9.2, pad=3, fontweight="normal")
        ax.set_xlabel("Time step")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        finish_axis(ax)
    ax_post_gt.set_ylabel(cfg["trajectory_y"], labelpad=5)
    ax_post_rc.tick_params(axis="y", labelleft=False)
    post_label = "(d)  Post-collapse extrapolation"
    add_panel_label(ax_post_gt, post_label, x=-0.13, y=1.16)

    fig.suptitle(cfg["label"], fontsize=11, fontweight="normal", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.085)
    return fig


def kuramoto_trajectory_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25), sharex=True, sharey=True)
    for row_index, (role, panel, condition) in enumerate(
        (("inter", "(a)", "Interpolation"), ("pre", "(b)", "Pre-sync extrapolation"))
    ):
        part = frame.loc[frame["role"] == role].sort_values("time").reset_index(drop=True)
        parameter = float(part["K"].iloc[0])
        for col_index, (series, title, color) in enumerate(
            (("gt", "GT", BLUE), ("scdt", "SCDT", RED))
        ):
            ax = axes[row_index, col_index]
            ax.plot(part["time"], part[series], color=color, lw=1.0)
            ax.axhline(
                0.5,
                color=REFERENCE,
                lw=0.8,
                linestyle=(0, (3, 2.5)),
                zorder=0,
            )
            ax.set_title(title, fontsize=9.2, pad=3, fontweight="normal")
            ax.set_xlim(part["time"].min(), part["time"].max())
            ax.set_ylim(-0.02, 1.02)
            finish_axis(ax)
        axes[row_index, 0].set_ylabel("r(t)")
        parameter_label = f"{parameter:.6f}" if role == "pre" else f"{parameter:.4f}"
        axes[row_index, 0].text(
            -0.13,
            1.13,
            f"{panel}  {condition},  K={parameter_label}",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=9.3,
        )
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.suptitle("Explosive Synchronization", fontsize=11, fontweight="normal", y=0.995)
    fig.subplots_adjust(
        left=0.09, right=0.985, top=0.88, bottom=0.12, wspace=0.10, hspace=0.43
    )
    return fig


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        stem.with_suffix(".jpg"),
        dpi=320,
        pil_kwargs={"quality": 95},
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scdt-code",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--kuramoto-data",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument(
        "--voltage-post-candidate",
        type=Path,
        default=None,
        help="Optional GT/SCDT candidate CSV used only for the Voltage post panels.",
    )
    parser.add_argument(
        "--voltage-post-steps",
        type=int,
        default=None,
        help="Optional number of Voltage post steps displayed in panel (d).",
    )
    args = parser.parse_args()
    configure_style()

    if args.voltage_post_candidate is None:
        packaged_candidate = (
            args.scdt_code
            / "data/voltage_active/post_200_candidate_trajectory.csv"
        )
        if packaged_candidate.exists():
            args.voltage_post_candidate = packaged_candidate
    if args.voltage_post_steps is None and args.voltage_post_candidate is not None:
        args.voltage_post_steps = 200

    if args.kuramoto_data is None:
        packaged = (
            args.scdt_code
            / "data/kuramoto_train_warmup_N2000/final_trajectory.csv"
        )
        workspace = (
            Path(__file__).resolve().parents[1]
            / "kuramoto_train_warmup_verify/nearest_warmup_N2000/w60/final_trajectory.csv"
        )
        args.kuramoto_data = packaged if packaged.exists() else workspace

    for system, cfg in SYSTEMS.items():
        cfg = dict(cfg)
        if system == "voltage_collapse" and args.voltage_post_steps is not None:
            cfg["post_steps"] = args.voltage_post_steps
        data_dir = args.scdt_code / "data" / cfg["data_dir"]
        trajectory = pd.read_csv(data_dir / "trajectory.csv")
        if system == "voltage_collapse" and args.voltage_post_candidate is not None:
            candidate = pd.read_csv(args.voltage_post_candidate).rename(
                columns={"scdt": "rc"}
            )
            candidate["role"] = "post"
            candidate = candidate[["role", "Q1", "time", "gt", "rc"]]
            trajectory = pd.concat(
                [trajectory.loc[trajectory["role"] != "post"], candidate],
                ignore_index=True,
            )
        reconstruction = load_reconstruction_data(args.scdt_code, system)
        figure = post_extended_trajectory_figure(trajectory, reconstruction, cfg)
        save(figure, args.output / f"{cfg['prefix']}2_trajectory")

    kuramoto = pd.read_csv(args.kuramoto_data)
    save(kuramoto_trajectory_figure(kuramoto), args.output / "K2_trajectory")


if __name__ == "__main__":
    main()
