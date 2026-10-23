#!/usr/bin/env python3
"""Plot simulated speedup against the OE 4x4 baseline as vector PDF."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "results.csv").open()))
sizes = (16, 32, 64, 128)
architectures = ("OE_2x5_LCU", "OE_5x5_LCU")
labels = {"OE_2x5_LCU": "OE 2x5 LCU (8 PEs)",
          "OE_5x5_LCU": "OE 5x5 LCU (20 PEs)"}

cycles = {(r["benchmark"], int(r["size"]), r["architecture"]):
          int(r["total_cycles"]) for r in rows}
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharey=True)
for axis, benchmark in zip(axes, ("MMUL", "GEMM")):
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                 label="OE 4x4 baseline")
    for architecture, marker in zip(architectures, ("o", "s")):
        speedups = [cycles[benchmark, n, "OE_4x4"] /
                    cycles[benchmark, n, architecture] for n in sizes]
        axis.plot(sizes, speedups, marker=marker, linewidth=1.6,
                  label=labels[architecture])
    axis.set_title(benchmark)
    axis.set_xlabel("Matrix dimension N")
    axis.set_xticks(sizes, labels=[str(n) for n in sizes])
    axis.grid(True, axis="y", alpha=0.25)
axes[0].set_ylabel("Speedup (OE 4x4 cycles / variant cycles)")
axes[1].legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(HERE / "heterogeneity_speedup.pdf", bbox_inches="tight")
