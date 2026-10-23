# LCU heterogeneity ablation

This folder contains the new row-LCU mappings and reproducible cycle data for
MMUL and GEMM. The original 4x4 kernels remain in their benchmark folders.
`gemm/instructions.csv` is the historical non-SRF GEMM instruction stream
restored from commit `9388c91`; no SRF variant is used.

## Run

From this directory:

```sh
python check_lcu_semantics.py
python run_all.py
python run_matched_controls.py
python plot_speedup.py
```

`run_all.py` executes the 24 requested combinations (two kernels, three main
architectures, four sizes) serially and writes `results.csv` and `results.json`.
It aborts on any incorrect output. `run_matched_controls.py` adds 16 cases
comparing each LCU mapping with the same number of compute PEs and an unrolled
no-LCU reference; it writes `matched_controls.csv/json`. The PDF plot shows
speedup over the original OE 4x4 mapping.

Inputs are generated with Python `random.Random(12345)` and values in
[-10, 10]. MMUL checks `C=A*B`; GEMM checks `C=125*(A*B)+14*C` with the
original C matrix. Output is compared element-by-element against an
independent CPU reference using signed 32-bit final results.

## Model semantics and limits

- `OE_2x5_LCU` has a 2x4 toroidal compute mesh and one LCU per row;
  `OE_5x5_LCU` has a 5x4 compute mesh and one LCU per row. The fifth physical
  column is a control endpoint, not an arithmetic PE or neighbor.
- LCUs use four local registers and `SADD`, `SSUB`, `BEQ`, `BNE`, `BLT`,
  `BGE`, and `JUMP`. The mapped K loop increments its LCU counter alongside
  an indirect load and branches alongside a PE pointer update. Each LCU can
  issue one op per PC; all row LCUs must agree on a branch target. The simulator
  rejects divergent row branches. Nested software loops are supported and
  checked by `check_lcu_semantics.py`.
- Only the K reduction is dynamic. Output tile traversal is statically
  configured. The 5-row design masks unused PEs in the last partial row tile.
- All matrix accesses use `LWI`/`SWI` indirect addressing and PE-local address
  registers. No SRF, VWR, scratchpad, or direct-stream addressing is used.
- PE memory and multiply timing retain this simulator's existing abstract
  costs (one plus the number of memory operations per PC; three cycles for a
  PC containing `SMUL`). Configuration cost is `static PCs * physical
  columns`. These are simulator cycles, not RTL or silicon cycles.
- The matched no-LCU control fully unrolls K but retains the same PE operation
  sequence and number of compute PEs. Its arithmetic/memory execution cycles
  equal the LCU version; the difference measures configuration compaction.
  It does not model a conventional PE branch implementation, which would
  contend for the four local registers already used by the kernel.
- This is a DISCO-inspired experiment, not a DISCO reproduction: DISCO's RTL
  has a 4x2 compute array with per-column LCUs and other coupled units, while
  these variants use the requested row organization and OE timing abstraction.
