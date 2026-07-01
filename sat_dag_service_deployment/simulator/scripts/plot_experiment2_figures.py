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
EXP2_DIR = ROOT_DIR / "simulator" / "outputs" / "exp2"
DEFAULT_AGG_CSV = EXP2_DIR / "aggregate" / "experiment2_main_comparison_M500_seed3.csv"
DEFAULT_FIG_DIR = EXP2_DIR / "figures"

NORMAL_SCENARIO = "normal_nominal"
STRESS_SCENARIO = "sparse_topology_stress"
ALGORITHMS = [
    "cpmv_dsd",
    "jsdts_aos_sat",
    "ondoc_sat",
    "floodsfcp_greedy",
    "greedy_resource",
]
DIAG_ALGORITHMS = ["cpmv_dsd", "jsdts_aos_sat", "ondoc_sat", "floodsfcp_greedy"]
DAG_TYPES = ["chain_like", "wide_shallow", "general"]

METHOD_LABELS = {
    "cpmv_dsd": "MV-\nDSD",
    "jsdts_aos_sat": "JSDTS-\nAOS",
    "ondoc_sat": "OnDoc-\nSAT",
    "floodsfcp_greedy": "Flood-\nChain",
    "greedy_resource": "Greedy-\nResource",
}
TABLE_LABELS = {
    "cpmv_dsd": "MV-DSD",
    "jsdts_aos_sat": "JSDTS-AOS",
    "ondoc_sat": "OnDoc-SAT",
    "floodsfcp_greedy": "Flood-Chain",
    "greedy_resource": "Greedy-Resource",
}
DAG_LABELS = {
    "chain_like": "Chain",
    "wide_shallow": "Wide",
    "general": "Mixed",
}
METHOD_COLORS = {
    "cpmv_dsd": "#2F5D7C",
    "jsdts_aos_sat": "#7B6EA8",
    "ondoc_sat": "#5B8C8A",
    "floodsfcp_greedy": "#C9A24A",
    "greedy_resource": "#B76E5B",
}
CP_COMPONENTS = [
    ("mean_cp_cmp_minutes_mean", "CP-Cmp", "#4F7CAC"),
    ("mean_cp_net_minutes_mean", "CP-Net", "#D29A5B"),
    ("mean_cp_idle_minutes_mean", "CP-Idle", "#6FAE9A"),
]
BAR_EDGE = "#6B7280"
ERROR_KW = {"ecolor": "#26323F", "lw": 0.85, "capsize": 2.0, "capthick": 0.85}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.6,
            "axes.labelsize": 7.6,
            "axes.titlesize": 7.6,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_for(
    rows: list[dict[str, str]],
    scenario: str,
    algorithm: str,
    group: str = "overall",
) -> dict[str, str] | None:
    return next(
        (
            row
            for row in rows
            if row.get("scenario") == scenario
            and row.get("algorithm") == algorithm
            and row.get("group", "overall") == group
        ),
        None,
    )


def value(row: dict[str, str] | None, column: str) -> float:
    if row is None:
        return math.nan
    raw = row.get(column, "")
    if raw == "":
        return math.nan
    return float(raw)


def ci95(row: dict[str, str] | None, std_column: str | None) -> float:
    if row is None or not std_column:
        return 0.0
    std = value(row, std_column)
    seed_count = value(row, "seed_count")
    if math.isnan(std) or math.isnan(seed_count) or seed_count <= 0:
        return 0.0
    return 1.96 * std / math.sqrt(seed_count)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.9, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#404040")
    ax.spines["bottom"].set_color("#404040")
    ax.tick_params(axis="both", length=2.6, color="#404040", pad=1.5)


def panel_label_below(ax: plt.Axes, label: str, y: float = -0.24) -> None:
    ax.set_xlabel(label, labelpad=0.8)


def save_figure(fig: plt.Figure, basename: str) -> list[Path]:
    DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        path = DEFAULT_FIG_DIR / f"{basename}.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.025)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_normal_summary_table(rows: list[dict[str, str]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.45, 1.38), constrained_layout=False)
    ax.axis("off")
    table_rows = []
    for algorithm in ALGORITHMS:
        row = row_for(rows, NORMAL_SCENARIO, algorithm)
        table_rows.append(
            [
                TABLE_LABELS[algorithm],
                f"{value(row, 'deadline_satisfaction_ratio_mean'):.3f}",
                f"{value(row, 'mean_makespan_minutes_mean'):.1f}",
                f"{value(row, 'p95_makespan_minutes_mean'):.1f}",
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Method", "DSR", "Avg. MS", "P95 MS"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.34, 0.20, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.12)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#404040")
        cell.set_linewidth(0.35)
        if row_idx == 0:
            cell.set_facecolor("#EDEDED")
            cell.set_text_props(weight="bold")
        elif col_idx == 0:
            color = METHOD_COLORS[ALGORITHMS[row_idx - 1]]
            cell.set_text_props(color=color, weight="bold")
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#FFFFFF")
    ax.text(0.5, -0.08, "Normal condition summary", transform=ax.transAxes, ha="center", va="top")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.98)
    return save_figure(fig, "fig4_exp2_literature_baselines_I")


def draw_metric_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    std_metric: str | None,
    ylabel: str,
    panel: str,
) -> None:
    x = np.arange(len(ALGORITHMS))
    vals = [value(row_for(rows, STRESS_SCENARIO, algorithm), metric) for algorithm in ALGORITHMS]
    errs = [ci95(row_for(rows, STRESS_SCENARIO, algorithm), std_metric) for algorithm in ALGORITHMS]
    ax.bar(
        x,
        vals,
        yerr=errs if any(err > 0 for err in errs) else None,
        width=0.62,
        color=[METHOD_COLORS[algorithm] for algorithm in ALGORITHMS],
        edgecolor=BAR_EDGE,
        linewidth=0.22,
        error_kw=ERROR_KW,
        zorder=3,
    )
    ax.set_ylabel(ylabel, labelpad=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    clean_axis(ax)
    panel_label_below(ax, panel, y=-0.27)
    ymax = max((v + e for v, e in zip(vals, errs) if not math.isnan(v)), default=1.0)
    if metric == "deadline_satisfaction_ratio_mean":
        ax.set_ylim(0, min(1.08, max(1.0, ymax * 1.16)))
    else:
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)


def plot_stress_main(rows: list[dict[str, str]]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.12), constrained_layout=False)
    specs = [
        ("deadline_satisfaction_ratio_mean", "deadline_satisfaction_ratio_std", "DSR", "(a) DSR"),
        ("p95_makespan_minutes_mean", "p95_makespan_minutes_std", "P95 Makespan (min)", "(b) P95 Makespan"),
        ("mean_makespan_minutes_mean", "mean_makespan_minutes_std", "Avg. Makespan (min)", "(c) Avg. Makespan"),
        ("mean_normalized_tardiness_mean", "mean_normalized_tardiness_std", "Norm. Tardiness", "(d) Tardiness"),
    ]
    for ax, spec in zip(axes.ravel(), specs):
        draw_metric_panel(ax, rows, *spec)
    fig.legend(
        [Patch(facecolor=METHOD_COLORS[algorithm], edgecolor=BAR_EDGE, linewidth=0.22) for algorithm in ALGORITHMS],
        [TABLE_LABELS[algorithm] for algorithm in ALGORITHMS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(ALGORITHMS),
        frameon=False,
        columnspacing=0.35,
        handlelength=0.75,
        handletextpad=0.20,
        borderaxespad=0.0,
        prop={"size": 6.2},
    )
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.06, top=0.91, wspace=0.48, hspace=0.50)
    return save_figure(fig, "fig5_exp2_literature_baselines_II")


def draw_cp_breakdown(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    x = np.arange(len(ALGORITHMS))
    bottoms = np.zeros(len(ALGORITHMS))
    handles = []
    for column, label, color in CP_COMPONENTS:
        vals = np.array([value(row_for(rows, STRESS_SCENARIO, algorithm), column) for algorithm in ALGORITHMS])
        vals = np.nan_to_num(vals, nan=0.0)
        bars = ax.bar(
            x,
            vals,
            width=0.62,
            bottom=bottoms,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.20,
            label=label,
            zorder=3,
        )
        handles.append(bars[0])
        bottoms += vals
    ax.set_ylabel("CP Component (min)", labelpad=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[algorithm] for algorithm in ALGORITHMS], rotation=0, ha="center")
    clean_axis(ax)
    panel_label_below(ax, "(a) CP Breakdown", y=-0.28)
    ax.set_ylim(0, float(np.max(bottoms)) * 1.18 if np.max(bottoms) > 0 else 1.0)
    ax.legend(
        handles,
        [label for _column, label, _color in CP_COMPONENTS],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.32,
        borderaxespad=0.0,
    )


def draw_dag_breakdown(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    x = np.arange(len(DAG_TYPES))
    width = 0.19
    offsets = (np.arange(len(DIAG_ALGORITHMS)) - (len(DIAG_ALGORITHMS) - 1) / 2.0) * width
    handles = []
    for idx, algorithm in enumerate(DIAG_ALGORITHMS):
        vals = []
        for dag_type in DAG_TYPES:
            baseline = value(row_for(rows, STRESS_SCENARIO, "cpmv_dsd", dag_type), "p95_makespan_minutes_mean")
            current = value(row_for(rows, STRESS_SCENARIO, algorithm, dag_type), "p95_makespan_minutes_mean")
            vals.append(current / baseline if baseline and math.isfinite(baseline) else math.nan)
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.90,
            color=METHOD_COLORS[algorithm],
            edgecolor=BAR_EDGE,
            linewidth=0.20,
            label=METHOD_LABELS[algorithm],
            zorder=3,
        )
        handles.append(bars[0])
    ax.axhline(1.0, color="#606060", linewidth=0.65, linestyle="--", zorder=2)
    ax.set_ylabel("Norm. P95 Makespan", labelpad=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([DAG_LABELS[item] for item in DAG_TYPES], rotation=0, ha="center")
    clean_axis(ax)
    panel_label_below(ax, "(b) DAG-Type P95", y=-0.28)
    ax.set_ylim(0.0, 2.55)
    ax.legend(
        handles,
        [TABLE_LABELS[item] for item in DIAG_ALGORITHMS],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=0.45,
        handlelength=0.9,
        handletextpad=0.30,
        borderaxespad=0.0,
    )


def plot_diagnosis(rows: list[dict[str, str]]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.86), constrained_layout=False)
    draw_cp_breakdown(axes[0], rows)
    draw_dag_breakdown(axes[1], rows)
    fig.subplots_adjust(left=0.17, right=0.995, bottom=0.23, top=0.86, wspace=0.46)
    return save_figure(fig, "fig6_exp2_performance_diagnosis")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Exp2 baseline comparison and diagnosis figures.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_AGG_CSV)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_style()
    rows = read_rows(args.input_csv)
    outputs: list[Path] = []
    outputs.extend(plot_normal_summary_table(rows))
    outputs.extend(plot_stress_main(rows))
    outputs.extend(plot_diagnosis(rows))
    print(f"Data file: {args.input_csv}")
    print("Generated figure files:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
