#!/usr/bin/env python3
"""Sequentially execute the 24 requested benchmark/configuration/size cases."""
import csv
import json
from pathlib import Path

from run_lcu_matrix import run_case

SIZES = (16, 32, 64, 128)
ARCHS = ("OE_4x4", "OE_2x5_LCU", "OE_5x5_LCU")
BENCHMARKS = ("mmul", "gemm")
OUT = Path(__file__).resolve().parent


def main():
    results = []
    for benchmark in BENCHMARKS:
        for size in SIZES:
            for architecture in ARCHS:
                result = run_case(benchmark, architecture, size)
                if not result["correct"]:
                    raise RuntimeError(f"Incorrect result: {result}")
                results.append(result)
                print(f"PASS {benchmark.upper()} {size} {architecture}: "
                      f"{result['total_cycles']} cycles", flush=True)
                (OUT / "results.json").write_text(json.dumps(results, indent=2))

    with (OUT / "results.csv").open("w", newline="") as handle:
        fields = ["benchmark", "size", "architecture", "correct", "written",
                  "total_outputs", "total_cycles", "execution_cycles",
                  "configuration_cycles", "pcs_executed", "lcu_operations",
                  "lcu_branch_operations", "lcu_taken_branches", "lcu_issue_pcs"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} validated cases to results.csv/results.json")


if __name__ == "__main__":
    main()
