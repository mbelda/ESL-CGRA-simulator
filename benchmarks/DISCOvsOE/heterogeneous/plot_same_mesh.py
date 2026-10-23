#!/usr/bin/env python3
"""Plot total modeled-cycle speedup for the preserved 4x4 compute mappings."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "same_mesh_results.csv").open()))
sizes = (16, 32, 64, 128)
architectures = ("OE_4x4_1LCU", "OE_4x5_4LCU")
labels = {"OE_4x4_1LCU": "4×4 + 1 global LCU",
          "OE_4x5_4LCU": "4×5 + 4 row LCUs"}
cycles = {(row["benchmark"], int(row["size"]), row["architecture"]):
          int(row["total_cycles"]) for row in rows}

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharey=True)
for axis, benchmark in zip(axes, ("MMUL", "GEM")):
    axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--",
                 label="OE 4×4 baseline")
    for architecture, marker, style in zip(architectures, ("o", "s"), ("-", "--")):
        speedups = [cycles[benchmark, size, "OE_4x4"] /
                    cycles[benchmark, size, architecture] for size in sizes]
        axis.plot(sizes, speedups, marker=marker, linewidth=1.6,
                  linestyle=style, markerfacecolor="white",
                  label=labels[architecture])
    axis.set_title(benchmark)
    axis.set_xlabel("Matrix dimension N")
    axis.set_xticks(sizes, labels=[str(size) for size in sizes])
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_ylim(0.9992, 1.00008)
axes[0].set_ylabel("Total-cycle speedup (baseline / variant)")
axes[1].legend(frameon=False, fontsize=8, loc="best")
fig.suptitle("Loop-control offload with the original 4×4 PE mappings", y=1.02)
fig.tight_layout()
fig.savefig(HERE / "same_mesh_speedup.pdf", bbox_inches="tight")
