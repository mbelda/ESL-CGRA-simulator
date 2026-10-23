#!/usr/bin/env python3
"""Plot validated SRF speedups for MMUL and GEMM from experiment CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SIZES = (16, 32, 64, 128)
VERSIONS = ("original", "srf_offsets", "srf_base")
SRF_VERSIONS = ("srf_offsets", "srf_base")
LABELS = {"srf_offsets": "SRF Offsets", "srf_base": "SRF Base"}
COLORS = {"srf_offsets": "#5E81AC", "srf_base": "#A3BE8C"}
MARKERS = {"srf_offsets": "s", "srf_base": "^"}


def read_results(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(size, version) for size in SIZES for version in VERSIONS}
    indexed = {}
    for row in rows:
        size = int(row["matrix_size"].split("x")[0])
        key = (size, row["version"])
        if key in indexed:
            raise ValueError(f"Duplicate experiment row in {path}: {key}")
        if row["validation"] != "PASS":
            raise ValueError(f"Unvalidated result in {path}: {row}")
        if row["correct_elements"] != row["total_elements"]:
            raise ValueError(f"Incomplete output validation in {path}: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"Expected all 12 validated size/version rows in {path}")

    for size in SIZES:
        hashes = {indexed[(size, version)]["input_sha256"] for version in VERSIONS}
        if len(hashes) != 1:
            raise ValueError(f"Variants used different inputs for size {size} in {path}")
    return indexed


def speedups(indexed):
    result = {version: [] for version in SRF_VERSIONS}
    for size in SIZES:
        baseline = int(indexed[(size, "original")]["total_simulated_cycles"])
        for version in SRF_VERSIONS:
            measured = int(indexed[(size, version)]["total_simulated_cycles"])
            result[version].append(baseline / measured)
    return result


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmul-data", type=Path,
        default=script_dir / "srf_ablation_results.csv",
    )
    parser.add_argument(
        "--gemm-data", type=Path,
        default=script_dir / "gemm_srf_ablation_results.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=script_dir / "Figures/srf_ablation.pdf",
    )
    args = parser.parse_args()
    data = {
        "MMUL": read_results(args.mmul_data),
        "GEMM": read_results(args.gemm_data),
    }

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.5), sharex=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.97, bottom=0.12, hspace=0.32)
    x = list(range(len(SIZES)))
    for ax, (kernel, rows), panel in zip(axes, data.items(), ("(a)", "(b)")):
        values = speedups(rows)
        for version in SRF_VERSIONS:
            ax.plot(
                x, values[version], label=LABELS[version],
                color=COLORS[version], marker=MARKERS[version],
                linewidth=1.7, markersize=4.5,
            )
        ax.axhline(1.0, color="#4C566A", linestyle="--", linewidth=0.9)
        ax.set_title(f"{panel} {kernel}", loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("Speedup vs. OE baseline")
        ax.set_ylim(0.999, 1.025)
        ax.set_yticks((1.000, 1.005, 1.010, 1.015, 1.020, 1.025))
        ax.grid(axis="y", color="#D8DEE9", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.2, loc="upper right")

    axes[-1].set_xticks(x, [f"{size} $\\times$ {size}" for size in SIZES])
    axes[-1].set_xlabel("Matrix dimensions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
