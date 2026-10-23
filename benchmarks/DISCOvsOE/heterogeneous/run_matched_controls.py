#!/usr/bin/env python3
"""Run same-compute PE controls with K statically unrolled and no LCU."""
import csv
import json
from pathlib import Path

from run_lcu_matrix import run_case

HERE = Path(__file__).resolve().parent
SIZES = (16, 32, 64, 128)
PAIRS = (("OE_2x5_LCU", "OE_2x4_REF"),
         ("OE_5x5_LCU", "OE_5x4_REF"))


def main():
    rows = []
    for size in SIZES:
        for benchmark in ("mmul", "gemm"):
            for lcu_arch, reference_arch in PAIRS:
                with_lcu = run_case(benchmark, lcu_arch, size)
                without_lcu = run_case(benchmark, reference_arch, size)
                if not with_lcu["correct"] or not without_lcu["correct"]:
                    raise RuntimeError(f"Incorrect matched control at N={size}: "
                                       f"{with_lcu} / {without_lcu}")
                if with_lcu["execution_cycles"] != without_lcu["execution_cycles"]:
                    raise RuntimeError("PE execution traces should have equal cycle costs")
                rows.append({
                    "benchmark": benchmark.upper(), "size": size,
                    "lcu_architecture": lcu_arch,
                    "reference_architecture": reference_arch,
                    "lcu_cycles": with_lcu["total_cycles"],
                    "reference_cycles": without_lcu["total_cycles"],
                    "lcu_execution_cycles": with_lcu["execution_cycles"],
                    "reference_execution_cycles": without_lcu["execution_cycles"],
                    "lcu_configuration_cycles": with_lcu["configuration_cycles"],
                    "reference_configuration_cycles": without_lcu["configuration_cycles"],
                    "configuration_cycles_saved": (without_lcu["configuration_cycles"] -
                                                     with_lcu["configuration_cycles"]),
                    "speedup": without_lcu["total_cycles"] / with_lcu["total_cycles"],
                    "correct": True,
                })
                print(f"PASS {benchmark.upper()} N={size} {lcu_arch} vs "
                      f"{reference_arch}: {rows[-1]['speedup']:.3f}x", flush=True)
                (HERE / "matched_controls.json").write_text(json.dumps(rows, indent=2))
    with (HERE / "matched_controls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matched-PE comparisons")


if __name__ == "__main__":
    main()
