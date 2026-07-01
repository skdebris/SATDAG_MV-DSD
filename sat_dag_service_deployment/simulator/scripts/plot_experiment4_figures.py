from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
EXP4_DIR = ROOT_DIR / "simulator" / "outputs" / "exp4"
AGG_DIR = EXP4_DIR / "aggregate"
FIG_DIR = EXP4_DIR / "figures"

SAMPLING_CSV = AGG_DIR / "exp4a_sampling_selection.csv"
SCALABILITY_CSV = AGG_DIR / "exp4b_scalability.csv"
KAPPA_CSV = AGG_DIR / "exp4c_kappa_sensitivity.csv"

SAMPLING_VARIANTS = ["full", "uniform_sampling", "naive_shapley"]
SCALING_VARIANTS = ["full", "uniform_sampling", "no_pruning"]

VARIANT_LABELS = {
    "full": "Full",
    "uniform_sampling": "Uniform",
    "naive_shapley": "Naive",
    "no_pruning": "No-Prune",
}
VARIANT_COLORS = {
    "full": "#2F5D7C",
    "uniform_sampling": "#5B8C8A",
    "naive_shapley": "#7B6EA8",
    "no_pruning": "#B76E5B",
}
VARIANT_MARKERS = {
    "full": "o",
    "uniform_sampling": "s",
    "naive_shapley": "^",
    "no_pruning": "D",
}
VARIANT_LINESTYLES = {
    "full": "-",
    "uniform_sampling": "-",
    "naive_shapley": "-.",
    "no_pruning": "--",
}
BAR_EDGE = "#6B7280"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.7,
            "axes.labelsize": 7.7,
            "axes.titlesize": 7.7,
            "legend.fontsize": 6.9,
            "xtick.labelsize": 6.8,
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


def rows_for(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("variant") == variant]


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.85, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#404040")
    ax.spines["bottom"].set_color("#404040")
    ax.tick_params(axis="both", length=2.6, color="#404040", pad=1.5)


def panel_label_below(ax: plt.Axes, text: str, y: float = -0.30) -> None:
    xlabel = ax.get_xlabel()
    ax.set_xlabel(f"{xlabel}\n{text}" if xlabel else text, labelpad=0.8)


def save_figure(fig: plt.Figure, basename: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        path = FIG_DIR / f"{basename}.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.025)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_lines_by_variant(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    variants: list[str],
    x_key: str,
    y_key: str,
    ylabel: str,
    scale_y: float = 1.0,
) -> list[plt.Line2D]:
    handles: list[plt.Line2D] = []
    for variant in variants:
        subset = sorted(rows_for(rows, variant), key=lambda row: value(row, x_key))
        x = np.array([value(row, x_key) for row in subset])
        y = np.array([value(row, y_key) * scale_y for row in subset])
        (line,) = ax.plot(
            x,
            y,
            marker=VARIANT_MARKERS[variant],
            markersize=3.4,
            linewidth=1.25,
            linestyle=VARIANT_LINESTYLES[variant],
            color=VARIANT_COLORS[variant],
            markeredgecolor="#26323F",
            markeredgewidth=0.35,
            label=VARIANT_LABELS[variant],
            zorder=3,
        )
        handles.append(line)
    ax.set_ylabel(ylabel)
    clean_axes(ax)
    return handles


def plot_oracle_per_sample(ax: plt.Axes, rows: list[dict[str, str]]) -> list[plt.Line2D]:
    handles: list[plt.Line2D] = []
    for variant in SAMPLING_VARIANTS:
        subset = sorted(rows_for(rows, variant), key=lambda row: value(row, "sample_size"))
        x = np.array([value(row, "sample_size") for row in subset])
        y = np.array([value(row, "oracle_calls_mean") / max(value(row, "sample_size"), 1.0) for row in subset])
        (line,) = ax.plot(
            x,
            y,
            marker=VARIANT_MARKERS[variant],
            markersize=3.4,
            linewidth=1.25,
            linestyle=VARIANT_LINESTYLES[variant],
            color=VARIANT_COLORS[variant],
            markeredgecolor="#26323F",
            markeredgewidth=0.35,
            label=VARIANT_LABELS[variant],
            zorder=3,
        )
        handles.append(line)
    ax.set_ylabel("Oracle Calls per Sample")
    clean_axes(ax)
    return handles


def plot_grouped_bars(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    variants: list[str],
    category_key: str,
    categories: list[float],
    y_getter,
    ylabel: str,
    xlabel: str,
    category_labels: list[str],
) -> list[plt.Rectangle]:
    x = np.arange(len(categories))
    width = min(0.22, 0.72 / max(len(variants), 1))
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2.0) * width
    handles: list[plt.Rectangle] = []
    for idx, variant in enumerate(variants):
        subset = {value(row, category_key): row for row in rows_for(rows, variant)}
        vals = [y_getter(subset.get(category), category) for category in categories]
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.90,
            color=VARIANT_COLORS[variant],
            edgecolor=BAR_EDGE,
            linewidth=0.22,
            label=VARIANT_LABELS[variant],
            zorder=3,
        )
        handles.append(bars[0])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel, labelpad=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    clean_axes(ax)
    ymax = max(
        (
            y_getter({value(row, category_key): row for row in rows_for(rows, variant)}.get(category), category)
            for variant in variants
            for category in categories
        ),
        default=1.0,
    )
    ax.set_ylim(0.0, ymax * 1.20 if ymax > 0 else 1.0)
    return handles


def plot_cost_pruning() -> list[Path]:
    scaling_rows = read_csv(SCALABILITY_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.35), constrained_layout=False)
    n_values = [40, 80, 120, 160]
    no_prune_calls = {
        value(row, "n_satellites"): max(value(row, "oracle_calls_mean"), 1.0)
        for row in rows_for(scaling_rows, "no_pruning")
    }
    handles = plot_grouped_bars(
        axes[0, 0],
        scaling_rows,
        SCALING_VARIANTS,
        "n_satellites",
        n_values,
        lambda row, _category: value(row, "planning_runtime_seconds_mean"),
        "Runtime (s)",
        "N",
        [str(item) for item in n_values],
    )
    panel_label_below(axes[0, 0], "(a) Runtime", y=-0.34)

    plot_grouped_bars(
        axes[0, 1],
        scaling_rows,
        SCALING_VARIANTS,
        "n_satellites",
        n_values,
        lambda row, _category: value(row, "oracle_calls_mean") / 1000.0,
        "Oracle Calls (k)",
        "N",
        [str(item) for item in n_values],
    )
    panel_label_below(axes[0, 1], "(b) Oracle Calls", y=-0.34)

    plot_grouped_bars(
        axes[1, 0],
        scaling_rows,
        SCALING_VARIANTS,
        "n_satellites",
        n_values,
        lambda row, category: value(row, "oracle_calls_mean") / no_prune_calls.get(category, 1.0) * 100.0,
        "Remaining Oracle Calls (%)",
        "N",
        [str(item) for item in n_values],
    )
    panel_label_below(axes[1, 0], "(c) Call Ratio", y=-0.34)

    plot_grouped_bars(
        axes[1, 1],
        scaling_rows,
        SCALING_VARIANTS,
        "n_satellites",
        n_values,
        lambda row, _category: value(row, "topk_jaccard"),
        "Top-K Stability",
        "N",
        [str(item) for item in n_values],
    )
    panel_label_below(axes[1, 1], "(d) Stability", y=-0.34)

    fig.legend(
        handles,
        [VARIANT_LABELS[item] for item in SCALING_VARIANTS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.70,
        handlelength=1.0,
        handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.15, right=0.995, bottom=0.08, top=0.89, wspace=0.47, hspace=0.62)
    return save_figure(fig, "fig10_exp4_computational_cost_pruning")


def plot_stability_quality() -> list[Path]:
    sampling_rows = read_csv(SAMPLING_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.88), constrained_layout=False)

    handles = plot_lines_by_variant(
        axes[0],
        sampling_rows,
        SAMPLING_VARIANTS,
        "sample_size",
        "contribution_variance",
        "Contribution Variance",
    )
    axes[0].set_xticks([50, 200, 500])
    axes[0].set_xticklabels(["50", "200", "500"])
    axes[0].set_xlabel("M", labelpad=0.8)
    panel_label_below(axes[0], "(a) Variance")

    plot_lines_by_variant(
        axes[1],
        sampling_rows,
        SAMPLING_VARIANTS,
        "sample_size",
        "topk_jaccard",
        "Top-K Jaccard Index",
    )
    axes[1].set_xticks([50, 200, 500])
    axes[1].set_xticklabels(["50", "200", "500"])
    axes[1].set_xlabel("M", labelpad=0.8)
    panel_label_below(axes[1], "(b) Top-K Stability")

    fig.legend(
        handles,
        [VARIANT_LABELS[item] for item in SAMPLING_VARIANTS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
        columnspacing=0.62,
        handlelength=1.0,
        handletextpad=0.28,
    )
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.23, top=0.76, wspace=0.40)
    return save_figure(fig, "fig11_exp4_estimation_stability_quality")


def main() -> None:
    setup_style()
    outputs: list[Path] = []
    outputs.extend(plot_cost_pruning())
    outputs.extend(plot_stability_quality())

    print("Data files:")
    for path in (SAMPLING_CSV, SCALABILITY_CSV, KAPPA_CSV):
        print(f"  {path}")
    print("Generated figure files:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
