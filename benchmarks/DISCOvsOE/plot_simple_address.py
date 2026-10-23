"""Plot speedup data from the validated Simple Address experiment JSON."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'simple_address_results.json').read_text())
sizes = [16, 32, 64, 128]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, name, title in zip(axes, ('mmul', 'gem'), ('MMUL', 'GEM')):
    rows = {row['size']: row for row in data if row['benchmark'] == name}
    speedups = [rows[size]['speedup'] for size in sizes]
    ax.plot(sizes, speedups, marker='o', linewidth=1.8, markersize=4)
    ax.axhline(1.0, color='0.5', linewidth=0.8, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Matrix dimension')
    ax.set_xticks(sizes, [str(x) for x in sizes])
    ax.grid(axis='y', alpha=0.25)
axes[0].set_ylabel('Speedup over OE original')
fig.tight_layout(pad=0.7)
fig.savefig(HERE / 'simple_address_speedup.pdf', bbox_inches='tight')
