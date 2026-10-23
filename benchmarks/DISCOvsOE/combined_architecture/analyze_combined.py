#!/usr/bin/env python3
"""Derive speedup, cycle savings, and additivity metrics from run_combined.py."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_ORDER = [
    "OE Baseline", "OE Simple Address", "OE + LCU global",
    "OE Simple Address + LCU global", "OE Simple Address + 4 LCUs",
]


def load(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["size"] = int(row["size"])
        row["total_cycles"] = int(row["total_cycles"])
        row["execution_cycles"] = int(row["execution_cycles"])
    return rows


def analyze(rows):
    by_key = {(r["benchmark"], r["size"], r["configuration"]): r for r in rows}
    out = []
    for benchmark in ("MMUL", "GEM"):
        sizes = sorted({r["size"] for r in rows if r["benchmark"] == benchmark})
        for size in sizes:
            baseline = by_key[benchmark, size, "OE Baseline"]["total_cycles"]
            sa = by_key[benchmark, size, "OE Simple Address"]["total_cycles"]
            lcu = by_key[benchmark, size, "OE + LCU global"]["total_cycles"]
            combo = by_key[benchmark, size, "OE Simple Address + LCU global"]["total_cycles"]
            combo4 = by_key[benchmark, size, "OE Simple Address + 4 LCUs"]["total_cycles"]
            sa_saving = baseline - sa
            lcu_saving = baseline - lcu
            combined_saving = baseline - combo
            combined4_saving = baseline - combo4
            additive = sa_saving + lcu_saving
            lcu_global_vs_four = combo - combo4
            for config in CONFIG_ORDER:
                item = by_key[benchmark, size, config]
                cycles = item["total_cycles"]
                out.append({
                    "benchmark": benchmark, "size": size, "configuration": config,
                    "total_cycles": cycles,
                    "speedup_vs_baseline": baseline / cycles,
                    "cycle_reduction_percent": 100 * (baseline - cycles) / baseline,
                    "cycles_saved_vs_simple_address": sa - cycles,
                    "cycles_saved_vs_lcu_global": lcu - cycles,
                    "simple_address_cycles_saved_alone": sa_saving,
                    "lcu_global_cycles_saved_alone": lcu_saving,
                    "combined_cycles_saved": combined_saving,
                    "additive_individual_savings": additive,
                    "combined_minus_additive_savings": combined_saving - additive,
                    "simple_address_plus_four_lcu_cycles_saved": combined4_saving,
                    "global_minus_four_lcu_cycles": lcu_global_vs_four,
                })
    return out


def write_comparison(rows, path):
    if not rows:
        raise ValueError("No experiment rows were loaded")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "combined_results.csv")
    parser.add_argument("--output", type=Path, default=HERE / "comparison_results.csv")
    args = parser.parse_args()
    write_comparison(analyze(load(args.input)), args.output)
    print(f"Wrote derived metrics to {args.output}")


if __name__ == "__main__":
    main()
