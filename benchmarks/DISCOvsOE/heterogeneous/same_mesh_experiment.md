# Same-mesh LCU experiment

This experiment isolates branch-control offload using the original manual
4x4 MMUL and GEM instruction streams. It is separate from `results.csv` and
`heterogeneity_speedup.pdf`, which document the earlier 2x5/5x5 remapped
experiment and remain unchanged.

## Reproduce

From the repository root:

```sh
python benchmarks/DISCOvsOE/heterogeneous/run_preserved_mesh.py --all
python benchmarks/DISCOvsOE/heterogeneous/plot_same_mesh.py
```

The runner executes MMUL and GEM at 16, 32, 64, and 128 using the same seed
and input matrices in every mode. It first validates complete dynamic PE
data/control traces at 4x4. Results are written to `same_mesh_results.csv`,
`same_mesh_results.json`, and `same_mesh_trace_validation.json`; the plot is
`same_mesh_speedup.pdf`.

## Control semantics

The PE array remains the original 4x4 `CGRA`. The 4x5 case adds four
sidecar control contexts and does not put them into the PE neighbour graph.
Only the three `BNE` instructions are replaced in their PE slots. The
original loop counters, bounds, initialization, and increments remain in
their original PE registers and instructions. The LCU reads the selected
predicate operands over an explicit control sideband, resolves the branch in
the same PC, and returns the original branch target to that PE's Rout to
preserve dataflow. In the four-LCU mode, the predicate is broadcast and all
four units must agree on the one global PC target. They therefore cannot
control independent row-local branches in this globally synchronized
program.

This is a functional simulator abstraction, not a confirmed connection in
the RTL. The RTL audit found a local four-register LCU datapath and a branch
path, but did not establish arbitrary PE-register-to-LCU operand or
LCU-to-PE-Rout paths. The experiment makes those interfaces explicit and
charges no extra execution cycles because branch evaluation is co-issued
with the PE PC. It charges one LCU configuration word for each of the three
static branch PCs; four row-LCU contexts are configured in parallel.

## Why the earlier results regressed

The earlier 2x5 and 5x5 schedules were newly generated kernels, not the
original manual mappings with only branches moved. The 2x5 version has eight
compute PEs and a two-row output tile, doubling the row-band tile count
relative to 4x4. The 5x5 version has twenty compute PEs and a five-row tile,
with partial rows masked at the edges. Both replaced the original routed,
manually packed schedule with per-tile pointer setup and a six-PC reduction
body (load A, load B, multiply, accumulate, advance A, advance B/branch).
The original 4x4 tile and operation placement were not preserved.

LCU operations themselves did not add waits: they were co-issued, with
branch evaluation in the same modeled PC. At N=16, for example, old MMUL
execution costs were 4,560 cycles for the original, 12,704 for 2x5, and
10,576 for 5x5. The generated mappings also triggered much larger modeled
memory costs: 2,602 cycles in the original versus 9,504 and 8,976. The
simulator charges one plus the total indirect memory operations in a PE PC,
so the generated multi-PE load PCs are expensive under this abstraction.
Configuration cost rose from 84 baseline cycles to 1,760 and 880. These
penalties come from the rewritten schedule, changed tile geometry, and
modelled memory/configuration costs, not from per-LCU synchronization stalls.

The older GEM baseline totals also contained a simulator accounting bug. A
taken branch changed `instr2exec` before `maxCyclesThisPC` measured the
current instruction's latency. At GEM PCs 27 and 30 the target is PC 2, so
the old code charged the 12-load latency of PC 1 instead of the one-cycle
branch PC. The corrected code measures latency using the saved current PC;
at N=16 this changes GEM execution from 5,415 to 5,235 cycles. The older
result files remain untouched; all numbers in this experiment use the
corrected accounting.

## Interpretation

The original control PCs are MMUL 8/16/19 and GEM 8/27/30. Each contains
non-control work on other RCs, so none is a removable control-only PC. The
LCU cases execute the same dynamic PCs and PE data operations, and produce
the same outputs. Execution cycles are consequently unchanged. One global
LCU and four synchronized row LCUs have the same modeled performance; the
four-LCU option performs four parallel predicate evaluations per branch
event and provides no latency advantage. The result is a zero-cycle
execution reduction for these schedules, not evidence that loop-control
offload could never improve a different mapping.
