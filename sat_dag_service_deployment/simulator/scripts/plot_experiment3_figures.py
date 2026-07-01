from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
EXP3_DIR = ROOT_DIR / "simulator" / "outputs" / "exp3"
AGG_DIR = EXP3_DIR / "aggregate"
FIG_DIR = EXP3_DIR / "figures"

OVERALL_CSV = AGG_DIR / "experiment3_overall_ablation_M500_seed3.csv"
DAGTYPE_CSV = AGG_DIR / "experiment3_dagtype_ablation_M500_seed3.csv"
BUDGET_CSV = AGG_DIR / "experiment3_sampling_budget_seed3.csv"

VARIANTS = ["full", "shapley", "no_struct", "no_strat"]
DIAG_VARIANTS = ["full", "shapley", "no_struct"]
BUDGET_VARIANTS = ["full", "no_strat"]
DAG_TYPES = ["chain_like", "wide_shallow", "general"]

VARIANT_LABELS = {
    "full": "Full",
    "shapley": "Shapley",
    "no_struct": "No-Struct",
    "no_strat": "No-Strat",
}

VARIANT_TICK_LABELS = {
    "full": "Full",
    "shapley": "Shapley",
    "no_struct": "No-\nStruct",
    "no_strat": "No-\nStrat",
}

DAG_LABELS = {
    "chain_like": "Chain",
    "wide_shallow": "Wide",
    "general": "Mixed",
}

COLORS = {
    "full": "#2F5D7C",
    "shapley": "#C9A24A",
    "no_struct": "#7B6EA8",
    "no_strat": "#5B8C8A",
    "cp_cmp": "#4F7CAC",
    "cp_net": "#D29A5B",
    "cp_idle": "#6FAE9A",
}
BAR_EDGE = "#6B7280"
ERROR_KW = {"ecolor": "#26323F", "lw": 0.85, "capsize": 2.0, "capthick": 0.85}

MARKERS = {
    "full": "o",
    "no_strat": "s",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.7,
            "axes.labelsize": 7.7,
            "axes.titlesize": 7.7,
            "legend.fontsize": 6.9,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 6.8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return float("nan")
    raw = row.get(key)
    if raw in (None, ""):
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def ci(row: dict[str, str] | None, metric_key: str) -> float:
    direct = value(row, f"{metric_key}_ci95")
    if math.isfinite(direct):
        return direct
    std = value(row, f"{metric_key}_std")
    seed_count = value(row, "seed_count")
    if math.isfinite(std) and math.isfinite(seed_count) and seed_count > 1:
        return 1.96 * std / math.sqrt(seed_count)
    return 0.0


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.85, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#404040")
    ax.spines["bottom"].set_color("#404040")
    ax.tick_params(axis="both", length=2.6, color="#404040", pad=1.5)


def panel_xlabel(ax: plt.Axes, text: str, prefix: str | None = None) -> None:
    label = text if prefix is None else f"{prefix}\n{text}"
    ax.set_xlabel(label, labelpad=0.8)


def save_figure(fig: plt.Figure, basename: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        path = FIG_DIR / f"{basename}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.025)
        outputs.append(path)
    plt.close(fig)
    return outputs


def row_by_variant(rows: list[dict[str, str]], variant: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("variant") == variant), None)


def plot_overall_ablation() -> list[Path]:
    rows = read_csv(OVERALL_CSV)
    metrics = [
        ("deadline_satisfaction_ratio_mean", "deadline_satisfaction_ratio", "DSR", "(a) DSR"),
        ("mean_normalized_tardiness_mean", "mean_normalized_tardiness", "Norm. Tardiness", "(b) Tardiness"),
        ("p95_makespan_minutes_mean", "p95_makespan_minutes", "P95 Makespan (min)", "(c) P95 Makespan"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.86), constrained_layout=False)
    x = np.arange(len(VARIANTS), dtype=float)

    handles = []
    for ax, (mean_key, ci_key, ylabel, label) in zip(axes, metrics):
        vals = [value(row_by_variant(rows, variant), mean_key) for variant in VARIANTS]
        yerr = [ci(row_by_variant(rows, variant), ci_key) for variant in VARIANTS]
        bars = ax.bar(
            x,
            vals,
            yerr=yerr,
            capsize=1.8,
            width=0.62,
            color=[COLORS[variant] for variant in VARIANTS],
            edgecolor=BAR_EDGE,
            linewidth=0.22,
            error_kw=ERROR_KW,
            zorder=3,
        )
        if not handles:
            handles = list(bars)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        clean_axes(ax)
        if "DSR" in ylabel:
            ax.set_ylim(0, min(1.05, max(vals) * 1.22 if vals else 1.0))
        panel_xlabel(ax, label)

    fig.legend(
        handles,
        [VARIANT_LABELS[item] for item in VARIANTS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(VARIANTS),
        frameon=False,
        columnspacing=0.55,
        handlelength=0.82,
        handletextpad=0.24,
        borderaxespad=0.0,
        prop={"size": 6.7},
    )
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.18, top=0.80, wspace=0.55)
    return save_figure(fig, "fig7_exp3_overall_ablation")


def plot_mechanism_diagnosis() -> list[Path]:
    overall_rows = read_csv(OVERALL_CSV)
    dag_rows = read_csv(DAGTYPE_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.84), constrained_layout=False)

    x = np.arange(len(VARIANTS), dtype=float)
    bottoms = np.zeros(len(VARIANTS))
    components = [
        ("mean_cp_cmp_minutes_mean", "CP-Cmp", COLORS["cp_cmp"]),
        ("mean_cp_net_minutes_mean", "CP-Net", COLORS["cp_net"]),
        ("mean_cp_idle_minutes_mean", "CP-Idle", COLORS["cp_idle"]),
    ]
    stack_handles = []
    for key, label, color in components:
        vals = np.array([value(row_by_variant(overall_rows, variant), key) for variant in VARIANTS])
        bars = axes[0].bar(
            x,
            vals,
            bottom=bottoms,
            width=0.62,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.20,
            label=label,
            zorder=3,
        )
        stack_handles.append(bars[0])
        bottoms += vals
    axes[0].set_ylabel("Critical-Path Component (min)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([VARIANT_TICK_LABELS[item] for item in VARIANTS])
    clean_axes(axes[0])
    panel_xlabel(axes[0], "(a) CP Breakdown")
    axes[0].legend(
        stack_handles,
        ["CP-Cmp", "CP-Net", "CP-Idle"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=0.35,
        handlelength=0.75,
        handletextpad=0.22,
        borderaxespad=0.0,
        prop={"size": 6.2},
    )

    dag_x = np.arange(len(DAG_TYPES), dtype=float)
    width = 0.22
    offsets = (np.arange(len(DIAG_VARIANTS)) - 1) * width
    base_by_dag: dict[str, float] = {}
    for dag_type in DAG_TYPES:
        base = next((row for row in dag_rows if row.get("dag_type") == dag_type and row.get("variant") == "full"), None)
        base_by_dag[dag_type] = max(value(base, "p95_makespan_minutes_mean"), 1e-9)

    bar_handles = []
    for idx, variant in enumerate(DIAG_VARIANTS):
        vals = []
        for dag_type in DAG_TYPES:
            row = next(
                (item for item in dag_rows if item.get("dag_type") == dag_type and item.get("variant") == variant),
                None,
            )
            vals.append(value(row, "p95_makespan_minutes_mean") / base_by_dag[dag_type])
        bars = axes[1].bar(
            dag_x + offsets[idx],
            vals,
            width=width * 0.9,
            color=COLORS[variant],
            edgecolor=BAR_EDGE,
            linewidth=0.20,
            label=VARIANT_LABELS[variant],
            zorder=3,
        )
        bar_handles.append(bars[0])
    axes[1].axhline(1.0, color="#606060", linewidth=0.6, linestyle="--", zorder=1)
    axes[1].set_ylabel("Norm. P95 Makespan")
    axes[1].set_xticks(dag_x)
    axes[1].set_xticklabels([DAG_LABELS[item] for item in DAG_TYPES])
    clean_axes(axes[1])
    panel_xlabel(axes[1], "(b) DAG-Type P95")
    axes[1].legend(
        bar_handles,
        [VARIANT_LABELS[item] for item in DIAG_VARIANTS],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=0.35,
        handlelength=0.75,
        handletextpad=0.22,
        borderaxespad=0.0,
        prop={"size": 6.2},
    )
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.27, top=0.78, wspace=0.45)
    return save_figure(fig, "fig8_exp3_mechanism_diagnosis")


def rows_for_variant(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    return sorted([row for row in rows if row.get("variant") == variant], key=lambda row: value(row, "sample_size"))


def plot_sampling_budget() -> list[Path]:
    rows = read_csv(BUDGET_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.72), constrained_layout=False)
    specs = [
        ("deadline_satisfaction_ratio_mean", "DSR", "(a) DSR"),
        ("p95_makespan_minutes_mean", "P95 Makespan (min)", "(b) P95 Makespan"),
    ]
    handles = []
    for ax, (metric, ylabel, label) in zip(axes, specs):
        for variant in BUDGET_VARIANTS:
            subset = rows_for_variant(rows, variant)
            x = [value(row, "sample_size") for row in subset]
            y = [value(row, metric) for row in subset]
            metric_base = metric.removesuffix("_mean")
            yerr = [ci(row, metric_base) for row in subset]
            (line,) = ax.plot(
                x,
                y,
                color=COLORS[variant],
                marker=MARKERS[variant],
                markersize=3.6,
                linewidth=1.25,
                markeredgecolor="#26323F",
                markeredgewidth=0.35,
                label=VARIANT_LABELS[variant],
                zorder=3,
            )
            lower = np.array(y) - np.array(yerr)
            upper = np.array(y) + np.array(yerr)
            ax.fill_between(x, lower, upper, color=COLORS[variant], alpha=0.12, linewidth=0.0, zorder=2)
            if len(handles) < len(BUDGET_VARIANTS):
                handles.append(line)
        ax.set_xlabel(f"M\n{label}", labelpad=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks([50, 100, 200, 300, 500])
        ax.set_xticklabels(["50", "100", "200", "300", "500"])
        clean_axes(ax)

    fig.legend(
        handles,
        [VARIANT_LABELS[item] for item in BUDGET_VARIANTS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.1,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.25, top=0.72, wspace=0.38)
    return save_figure(fig, "fig9_exp3_sampling_budget")


def main() -> None:
    setup_style()
    outputs: list[Path] = []
    outputs.extend(plot_overall_ablation())
    outputs.extend(plot_mechanism_diagnosis())
    outputs.extend(plot_sampling_budget())

    print("Data files:")
    for path in (OVERALL_CSV, DAGTYPE_CSV, BUDGET_CSV):
        print(f"  {path}")
    print("Generated figure files:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
