# Simple Address + LCU ablation

This experiment combines the validated Simple Address MMUL and GEM instruction
streams with the control semantics from `heterogeneous/run_preserved_mesh.py`.
The 16 compute PEs retain their logical 4×4 toroidal mesh, 4×4 output tiles,
local registers, arithmetic instructions, and memory issue order. A global
LCU or four synchronized row LCUs handle the conditional loop branches.
The four-LCU mode represents a 4×5 arrangement, with a fifth control
endpoint per row; the simulator keeps the compute network at 4×4. The
LCU contexts do not participate in PE neighbor routing.

## Reproduce

Run from the repository root with the Python environment used by `cgra.py`:

```sh
python benchmarks/DISCOvsOE/combined_architecture/run_combined.py
python benchmarks/DISCOvsOE/combined_architecture/analyze_combined.py
python benchmarks/DISCOvsOE/combined_architecture/plot_combined.py
```

`run_combined.py` accepts `--sizes 16 32 64 128` and `--output-dir PATH`.
The default run executes five configurations for both MMUL and GEM at every
size. Runs are sequential and each simulator gets its own memory image and
direct-address registers. The inputs use Python seed 12345 with 32-bit
integer arithmetic; GEM computes `C = 125*A*B + 14*C_original`.

The main artifacts are:

- `combined_results.csv` and `.json`: measured execution, configuration,
  and total cycles, correctness, instruction version, and control/memory data.
- `direct_trace_validation.json`: load/store sequence, pointer endpoint,
  and per-loop branch-count checks for each size.
- `comparison_results.csv`: speedups, percentage reductions, savings relative
  to each individual optimization, and additivity measurements.
- `combined_speedup.pdf` and `combined_cycle_savings.pdf`: vector figures
  generated from the measured CSV.
- `mmul_instructions_simple_address_lcu.csv` and
  `gem_instructions_simple_address_lcu.csv`: compacted PE instruction streams.
  They are generated from the validated Simple Address CSVs. The accompanying
  relocated LCU branches are specified by `compile_lcu_program()` in the
  runner; an LCU run must use that control schedule. A retained branch in the
  MMUL CSV is replaced by `NOP` in the PE at load time and co-issued by the
  LCU.

## Schedule and cycle accounting

Simple Address removes original address PCs 10 and 11 in MMUL, and 10, 11,
22, and 23 in GEM. Once their address operations are gone, MMUL Simple Address
PCs 8 and 17 and GEM Simple Address PCs 8, 23, and 26 contain only branches.
The combined schedule removes those PCs and resolves each branch after its
preceding PE PC. The other MMUL branch remains co-issued at its mixed
data/control PC. Branch targets are remapped after compaction; an explicit
sideband reproduces the removed branch PE's `Rout` value before the next PE
PC. The original loop counters and bounds remain in the same local PE
registers. The compiler checks that no relocated branch needs a register
value produced by its preceding PE PC; the LCU reads its operands at that
PC's start, preserving the earlier control semantics without forwarding.
No arithmetic, matrix loads/stores, or tiling are redesigned.

The simulator charges one plus the number of memory operations for a PC with
memory, three cycles for a PC containing multiplication, and one cycle for
other PCs. A branch evaluated by an LCU at an existing PC boundary adds no
issue latency. Total cycles include four PE configuration cycles per retained
PC and one LCU configuration cycle per distinct branch PC; the four row LCUs
are configured in parallel. Host-side stream tracing/reordering and matrix
reference calculation are excluded from kernel cycles.

All five configurations are rerun on the current simulator. Historical GEM
cycle counts in the imported Simple Address result files predate the fix in
`cgra.py` that charges a taken branch to the PC actually issued; MMUL cycle
counts are unchanged. `combined_results.csv` is the consistent source for
the five-way comparison.

## Validation and interpretation

Every run checks the complete output matrix against an independent Python
reference. Direct loads are checked against the original kernel's indirect
load trace, including the configuration words; direct stores are checked
against their planned logical destinations and complete `C` output. Each
column's final load/store pointer is checked against the number of issued
accesses. The three Simple Address configurations must have identical
direct-access traces and identical taken/total branch counts at every
original Simple Address branch PC. The simulator also checks agreement among
four LCUs on every branch target. All 40 reported runs passed.

The cycle benefit is schedule dependent. The LCU alone cannot remove any of
the original three branch PCs, because those PCs also issue address work.
Simple Address turns several of them into branch-only PCs, so combining the
features saves more than the sum of the individual cycle savings. One LCU and
four LCUs have equal cycles and branch outcomes because these loops are
synchronized and the model has one global PC. The model does not establish a
physical PE-register-to-LCU path or Rout sideband in RTL. The streamed memory
layout replicates operands, and its host-side preparation and extra capacity
are outside the cycle count. No SRF, VWR, or scratchpad is used.
