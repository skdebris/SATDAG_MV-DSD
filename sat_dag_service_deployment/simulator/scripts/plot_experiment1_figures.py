from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT_DIR = Path(__file__).resolve().parents[2]
EXP1_DIR = ROOT_DIR / "simulator" / "outputs" / "exp1"
DEFAULT_AGG_CSV = EXP1_DIR / "aggregate" / "experiment1_architectural_advantage_M500_seed3.csv"
DEFAULT_FIG_DIR = EXP1_DIR / "figures"

SCENARIO = "sparse_topology_stress"
ALGORITHMS = ["cpmv_dsd", "dependency_blind", "sfc_path_decomp"]
METHOD_LABELS = {
    "cpmv_dsd": "MV-\nDSD",
    "dependency_blind": "Dep.-\nBlind",
    "sfc_path_decomp": "Chain-\nStyle",
}
LEGEND_LABELS = {
    "cpmv_dsd": "MV-DSD",
    "dependency_blind": "Dep.-Blind",
    "sfc_path_decomp": "Chain-Style",
}
METHOD_COLORS = {
    "cpmv_dsd": "#2F5D7C",
    "dependency_blind": "#B76E5B",
    "sfc_path_decomp": "#5B8C8A",
}
BAR_EDGE = "#6B7280"
ERROR_KW = {"ecolor": "#26323F", "lw": 0.85, "capsize": 2.0, "capthick": 0.85}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.7,
            "axes.labelsize": 7.7,
            "axes.titlesize": 7.8,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("scenario") == SCENARIO and row.get("group") == "overall"
    ]
    missing = [algorithm for algorithm in ALGORITHMS if not any(row.get("algorithm") == algorithm for row in selected)]
    if missing:
        raise ValueError(f"Missing Exp1 aggregate rows for algorithms: {missing}")
    return selected


def row_for(rows: list[dict[str, str]], algorithm: str) -> dict[str, str]:
    for row in rows:
        if row.get("algorithm") == algorithm:
            return row
    raise KeyError(algorithm)


def value(row: dict[str, str], column: str) -> float:
    raw = row.get(column, "")
    if raw == "":
        return math.nan
    return float(raw)


def ci95(row: dict[str, str], std_column: str | None) -> float:
    if not std_column or std_column not in row:
        return 0.0
    std = value(row, std_column)
    seed_count = value(row, "seed_count") if "seed_count" in row else math.nan
    if math.isnan(std) or math.isnan(seed_count) or seed_count <= 0:
        return 0.0
    return 1.96 * std / math.sqrt(seed_count)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.9, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#404040")
    ax.spines["bottom"].set_color("#404040")
    ax.tick_params(axis="both", length=2.8, color="#404040", pad=1.5)


def panel_label_below(ax: plt.Axes, label: str) -> None:
    ax.set_xlabel(label, labelpad=0.8)


def annotate(ax: plt.Axes, bars, fmt: str, yerr: list[float]) -> None:
    heights = [bar.get_height() for bar in bars]
    max_height = max([h for h in heights if not math.isnan(h)] or [1.0])
    offset = max_height * 0.035
    for bar, err in zip(bars, yerr):
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + err + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#202020",
            clip_on=False,
        )


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric_column: str,
    std_column: str | None,
    title: str,
    ylabel: str,
    fmt: str,
) -> None:
    x = np.arange(len(ALGORITHMS))
    values = [value(row_for(rows, algorithm), metric_column) for algorithm in ALGORITHMS]
    errors = [ci95(row_for(rows, algorithm), std_column) for algorithm in ALGORITHMS]
    ax.bar(
        x,
        values,
        yerr=errors if any(error > 0 for error in errors) else None,
        width=0.62,
        color=[METHOD_COLORS[algorithm] for algorithm in ALGORITHMS],
        edgecolor=BAR_EDGE,
        linewidth=0.22,
        error_kw=ERROR_KW,
        zorder=3,
    )
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    clean_axis(ax)
    panel_label_below(ax, title)
    ymax = max((v + e for v, e in zip(values, errors) if not math.isnan(v)), default=1.0)
    if metric_column in {"tcr_mean", "deadline_satisfaction_ratio_mean"}:
        ax.set_ylim(0.0, min(1.22, max(1.02, ymax * 1.20)))
    else:
        ax.set_ylim(0.0, ymax * 1.20 if ymax > 0 else 1.0)


def save_figure(fig: plt.Figure, fig_dir: Path, basename: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "png"):
        path = fig_dir / f"{basename}.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.025)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_exp1(rows: list[dict[str, str]], fig_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.62), constrained_layout=False)
    draw_panel(
        axes[0],
        rows,
        "deadline_satisfaction_ratio_mean",
        "deadline_satisfaction_ratio_std",
        "(a) DSR",
        "DSR",
        "{:.3f}",
    )
    draw_panel(
        axes[1],
        rows,
        "mean_makespan_minutes_mean",
        "mean_makespan_minutes_std",
        "(b) Avg. Makespan",
        "Avg. Makespan (min)",
        "{:.1f}",
    )
    draw_panel(
        axes[2],
        rows,
        "p95_makespan_minutes_mean",
        None,
        "(c) P95 Makespan",
        "P95 Makespan (min)",
        "{:.1f}",
    )
    fig.legend(
        [Patch(facecolor=METHOD_COLORS[algorithm], edgecolor=BAR_EDGE, linewidth=0.22) for algorithm in ALGORITHMS],
        [LEGEND_LABELS[algorithm] for algorithm in ALGORITHMS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(ALGORITHMS),
        frameon=False,
        columnspacing=0.8,
        handlelength=0.9,
        handletextpad=0.28,
        borderaxespad=0.0,
        prop={"size": 6.9},
    )
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.15, top=0.87, wspace=0.52)
    return save_figure(fig, fig_dir, "fig3_exp1_architectural_advantage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Exp1 fixed-scene architectural advantage figure.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_AGG_CSV)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_style()
    rows = load_rows(args.input_csv)
    outputs = plot_exp1(rows, args.figure_dir)
    print(f"Data file: {args.input_csv}")
    print("Generated figure files:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
