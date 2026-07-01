from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
EXP5_DIR = ROOT_DIR / "simulator" / "outputs" / "exp5"
DEFAULT_AGG_CSV = EXP5_DIR / "aggregate" / "experiment5_cvar_sweep_M500_seed3.csv"
DEFAULT_FIG_DIR = EXP5_DIR / "figures"

LINE_COLOR = "#2F5D7C"
MARKER_FACE = "#C9A24A"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.7,
            "axes.labelsize": 7.7,
            "axes.titlesize": 7.7,
            "legend.fontsize": 6.9,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 6.8,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: value(row, "lambda_cvar"))


def value(row: dict[str, str], key: str) -> float:
    raw = row.get(key)
    if raw in (None, ""):
        return float("nan")
    return float(raw)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.9, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#404040")
    ax.spines["bottom"].set_color("#404040")
    ax.tick_params(axis="both", length=2.8, color="#404040", pad=1.5)


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    panel: str,
    show_recommended: bool = False,
) -> None:
    x = np.array([value(row, "lambda_cvar") for row in rows])
    y = np.array([value(row, metric) for row in rows])
    if show_recommended:
        ax.axvspan(0.5, 0.7, color="#E8E8E8", zorder=0)
        ax.text(
            0.60,
            0.96,
            "Rec. range",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.6,
            color="#505050",
        )
    ax.plot(
        x,
        y,
        color=LINE_COLOR,
        marker="o",
        markersize=3.6,
        linewidth=1.25,
        markerfacecolor=MARKER_FACE,
        markeredgecolor="#26323F",
        markeredgewidth=0.35,
        zorder=3,
    )
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.set_xlabel(r"Risk Weight $\lambda$" + f"\n{panel}", labelpad=0.8)
    ax.set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    ax.set_xticklabels(["0", ".1", ".3", ".5", ".7", "1.0"])
    clean_axis(ax)
    if metric == "deadline_satisfaction_ratio_mean":
        ax.set_ylim(max(0.0, np.nanmin(y) - 0.04), min(1.02, np.nanmax(y) + 0.03))


def save_figure(fig: plt.Figure, fig_dir: Path, basename: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        path = fig_dir / f"{basename}.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.025)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_exp5(rows: list[dict[str, str]], fig_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.52), constrained_layout=False)
    draw_panel(axes[0], rows, "deadline_satisfaction_ratio_mean", "DSR", "(a) DSR")
    draw_panel(axes[1], rows, "p95_makespan_minutes_mean", "P95 Makespan (min)", "(b) P95", True)
    draw_panel(
        axes[2],
        rows,
        "cvar_makespan_minutes_mean",
        r"CVaR$_{0.90}$ Makespan (min)",
        "(c) CVaR",
        True,
    )
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.31, top=0.94, wspace=0.54)
    return save_figure(fig, fig_dir, "fig12_exp5_cvar_risk_extension")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Exp5 CVaR-aware risk extension figure.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_AGG_CSV)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_style()
    rows = read_rows(args.input_csv)
    outputs = plot_exp5(rows, args.figure_dir)
    print(f"Data file: {args.input_csv}")
    print("Generated figure files:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
