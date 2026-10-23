#!/usr/bin/env python3
"""Plot the combined ablation from the measured CSV (no embedded data)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOT_CONFIGS = [
    ("OE Simple Address", "Simple Address", "#4C78A8", "o"),
    ("OE + LCU global", "LCU global", "#F58518", "s"),
    ("OE Simple Address + LCU global", "Simple Address + LCU", "#54A24B", "^"),
    ("OE Simple Address + 4 LCUs", "Simple Address + 4 LCUs", "#B279A2", "D"),
]


def read_data(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["size"] = int(row["size"])
        row["total_cycles"] = int(row["total_cycles"])
    return rows


def plot(rows, output_dir):
    lookup = {(r["benchmark"], r["size"], r["configuration"]): r["total_cycles"]
              for r in rows}
    sizes = sorted({r["size"] for r in rows})
    benchmarks = ("MMUL", "GEM")
    ratios = {}
    for benchmark in benchmarks:
        for label, _, _, _ in PLOT_CONFIGS:
            ratios[benchmark, label] = [
                lookup[benchmark, n, "OE Baseline"] /
                lookup[benchmark, n, label] for n in sizes]
    low = min(min(values) for values in ratios.values())
    high = max(max(values) for values in ratios.values())
    pad = max(0.006, (high - low) * 0.12)
    ymin, ymax = low - pad, high + pad

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, benchmark in zip(axes, benchmarks):
        for key, label, color, marker in PLOT_CONFIGS:
            four_lcus = key.endswith("4 LCUs")
            ax.plot(sizes, ratios[benchmark, key], label=label, color=color,
                    marker=marker, linewidth=1.8 if not four_lcus else 0,
                    markersize=8 if four_lcus else 5,
                    markerfacecolor="white" if four_lcus else color,
                    markeredgewidth=1.5 if four_lcus else 1.0,
                    zorder=3 if four_lcus else 4)
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0,
                   label="OE Baseline" if benchmark == "MMUL" else None)
        ax.set_title(benchmark)
        ax.set_xlabel("Matrix dimension $N$ ($N\\times N$)")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Total-cycle speedup over OE Baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = output_dir / "combined_speedup.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    savings = {}
    for benchmark in benchmarks:
        baseline = [lookup[benchmark, n, "OE Baseline"] for n in sizes]
        for key, _, _, _ in PLOT_CONFIGS:
            savings[benchmark, key] = [
                base - lookup[benchmark, n, key]
                for n, base in zip(sizes, baseline)]
    for ax, benchmark in zip(axes, benchmarks):
        for key, label, color, marker in PLOT_CONFIGS:
            four_lcus = key.endswith("4 LCUs")
            ax.plot(sizes, savings[benchmark, key], label=label, color=color,
                    marker=marker, linewidth=1.8 if not four_lcus else 0,
                    markersize=8 if four_lcus else 5,
                    markerfacecolor="white" if four_lcus else color,
                    markeredgewidth=1.5 if four_lcus else 1.0,
                    zorder=3 if four_lcus else 4)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(benchmark)
        ax.set_xlabel("Matrix dimension $N$ ($N\\times N$)")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cycles saved relative to OE Baseline")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savings_path = output_dir / "combined_cycle_savings.pdf"
    fig.savefig(savings_path, bbox_inches="tight")
    plt.close(fig)
    return path, savings_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "combined_results.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in plot(read_data(args.input), args.output_dir):
        print(f"Wrote vector PDF {path}")


if __name__ == "__main__":
    main()
