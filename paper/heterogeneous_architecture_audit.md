# RTL audit: OpenEdge and DISCO

This note records the RTL review used to scope the heterogeneous ablation. The RTL repositories were inspected read-only. The paths below are the local checkouts found in this environment; they differ from the path in the request.

## Repositories and revision state

| Project | Local path | State observed |
|---|---|---|
| ESL CGRA Simulator | `/home/mbelda/Documents/GitHub/mbelda/ESL-CGRA-simulator` | Branch `DISCOvsOE-Ablation-Heterogeneous`, based on `af54c21` (`Merge branch 'Compigra-Kernel'`). The user-provided, untracked `paper/Ablation.tex` and submission PDF are preserved. |
| Hippsilon / OpenEdge | `/home/mbelda/Documents/GitHub/mbelda/HEEPsilon` | `main`, `f8b3cd2`, clean. |
| HIPA Live / DISCO | `/home/mbelda/Documents/GitHub/mbelda/heepalive-imec` | `mmul`, `da1988e`, pre-existing dirty application/generated files. RTL files reviewed are read-only; matching RTL was checked in the second checkout under `orig-heepalive/heepalive-imec`. |

`benchmarks/DISCOvsOE/gem` and `benchmarks/DISCOvsOE/gemm` contain only `__pycache__` in this checkout; no source runner or instruction CSV is tracked here. MMUL source files are present. This means the prior DISCOvsOE GEM experiment cannot be reproduced from this branch without supplying or reconstructing its inputs.

## OpenEdge (Hippsilon)

OpenEdge's CGRA RTL is under `hw/vendor/esl_epfl_cgra/rtl`; HIP/X-HEEP integration is outside that core. `cgra_pkg.sv:12-16` fixes the instantiated array at 4 rows by 4 columns. `cgra_pkg.sv:28-37` defines 32-bit datapath/instructions and four local registers per RC; `:39-52` defines the context-memory organization. The fixed package constants are not a run-time array-size parameter.

`reconfigurable_cell.sv:46-59` gates configuration and datapath execution. `datapath.sv:69-77,216-248` selects two operands from zero, self, four cardinal neighbors, local registers, or an immediate, then dispatches the opcode. `reg_file.sv:26-43` is a four-entry, 32-bit local RF with asynchronous reads and synchronous writes. `cgra_pkg.sv:90-117` lists arithmetic, conditional selection, branches/jumps, direct/indirect loads/stores, and exit operations.

`cgra_rcs.sv:363-393,413-450` wires registered RC outputs and flags to cardinal neighbors with wrap-around. Thus neighbor data consumed by a PE is from registered prior-cycle state. The array executes a configured context under controller/synchronizer coordination; the PC and memory handshakes are implemented in `program_counter.sv`, `cgra_controller.sv`, `synchronizer.sv`, and `data_bus_handler.sv`. `data_bus_handler.sv:78-102,129-137` keeps per-column direct read/write pointers initialized by the host and advances a successful direct access by a fixed 4-byte word. Indirect accesses use the instruction-supplied address. The column memory path has arbitration and request/response handshakes; it is not an instantaneous Python list lookup.

There is no column-shared scalar register file in this OpenEdge RTL. The SoC CPU, bus, interrupts, and host loading/configuration are system integration, not RC datapath features. Multiplication merits a fidelity caveat: the controller contains a three-cycle multiplier-stall path (`cgra_controller.sv:188-204`), but the inspected `alu.sv:427,430` drives the stall output inactive and multiplication is combinational (`alu.sv:148-150`) in this revision.

## DISCO (HIPA Live)

The DISCO RTL is in `hw/vendor/disco/rtl`; integration is in `disco_top.sv` and `disco_top_wrapper.sv`. The package used by the simulation config (`hw/vendor/disco/sim/cgra_pkg.sv:19-28,48-53,157-160,181-183`) describes a 4-row by 2-column compute array, two cores, a 128-word-wide vector register line, eight scalar-register words, and a scratchpad with 64 lines. This is materially different from a 2-by-7 PE grid.

DISCO's recurring column organization consists of RCs plus a Loop Control Unit (LCU), Load/Store Unit (LSU), and Multi-context eXecution Unit (MXCU) per column. `lcu_top.sv:85-106` and `lcu_datapath.sv:71-100,125-143,173-186` implement the LCU datapath and PC-control operations. The LCU has a four-entry local register file and ordinary ALU instructions; it has no dedicated loop-counter opcode or hardware counter. Software forms loops using arithmetic updates and conditional branches. `lcu_alu.sv:182-218` implements the branch comparisons. `lcu_top.sv:66-83` has a branch-address selection explicitly wired for two columns, an important scaling assumption.

`lsu_top.sv:65-108` and `lsu_cell.sv:90-120` implement scratchpad transactions with grant-based stalls. Scratchpad line loads update scalar-register and VWR state (`lsu_cell.sv:112-118,246-261`). `scratchpad_memory.sv:79-102,114-161` arbitrates DMA and column masters; at most one read and one write are granted per cycle, independently. The DMA (`cgra_dma.sv`) moves data between the scratchpad and external TCDM. Thus LSU behavior is coupled to the SPM, line granularity, VWRs, and scalar RF; an isolated scalar-memory unit is an explicit abstraction, not the DISCO LSU.

The MXCU (`mxcu_top.sv:47-75`, `mxcu_cell.sv:62-82,96-169`, `mxcu_datapath.sv:74-84`) configures/selects and latches VWR slices across RC rows, controls VWR indexes/masks, and has paths to update the scalar RF from LCU, RC, MXCU, or LSU sources. It is not a free-standing arithmetic helper. VWR latches and conflict checks are in `vwr_latch.sv:45-79`; scalar RF write collision checks are in `scalar_reg_file_mux.sv:41-63`. Therefore VWR, SRF, MXCU, LSU, and SPM effects cannot all be claimed as independent without additional ablations.

## Comparison and Python-model fidelity

| Feature | OpenEdge RTL | DISCO RTL | `cgra.py` on this branch | Consequence |
|---|---|---|---|---|
| Compute array | Fixed 4x4 RC grid in package | 4x2 RC grid plus per-column specialist units | Fixed 4x4 constants (`cgra.py:8-12,60-68`) | Good OE shape match only for 4x4; neither hardware shape is dynamically selectable here. |
| PE state | Four local 32-bit registers | RC local state plus shared/control state | Four integer local registers (`PE`, `cgra.py:211+`) | Partial: width/overflow behavior can differ by operation. |
| Neighbor network | Registered cardinal outputs with wraparound | RC array plus VWR/MXCU cross-row selection and control paths | Toroidal cardinal access to `old_out` (`cgra.py:158-173`) | The torus and prior-output timing correspond at the basic RC-neighbor level. |
| Instruction timing | PC holds on real grant/stall conditions | Global/per-column synchronization and LSU stalls | One Python step per PC; estimated delay from opcode counts (`cgra.py:104-130,132-156`) | Abstract timing; not RTL-cycle-accurate. |
| Direct memory | Per-column direct pointer, fixed +4 on successful access | LSU/SPM transactions, line transfers and grants | `LWD/SWD` use instruction increment (`cgra.py:175-201, PE.run_instr`) and direct Python memory lookup | Behavioral mismatch: pointer increments differ; arbitration and wait states are omitted. |
| Indirect memory | Address-bearing instruction through bus handshake | LSU/SPM address and grant path | Immediate list lookup/store | Correctness-level abstraction may be useful, cycle count is not hardware latency. |
| Multiply | Combinational in inspected ALU revision; stall signal inactive | Separate DISCO RC ALU implementation | `SMUL` charged 3 modeled cycles (`cgra.py:116-122`) | Potentially significant cycle-count mismatch for multiply-heavy kernels. |
| Shared scalar file | Absent | Eight words per column in sim package | Absent on this base branch | SRF ablation is not represented by stock OE; experimental extension only. |
| Control | RC instructions and PC machinery | LCU has independent loop/branch datapath and PC control | Branches are issued by RC programs | LCU-like loop offload is a meaningful controlled extension. |
| Memory/vector subsystem | Per-column bus handler, no DISCO SPM/VWR | LSU + SPM + VWR + SRF + DMA | Flat Python address/value rows | Comparing an isolated scalar LSU tests a deliberately narrow abstraction, not DISCO memory as a whole. |

The Python model is useful as an architectural functional simulator for a subset of OE scheduling, but its accumulated cycle estimates should not be called RTL-accurate. Most important are (1) direct pointer increments, (2) omitted per-column arbitration and memory backpressure, (3) global row-major Python effects where RTL uses synchronized hardware state, and (4) the three-cycle modeled multiply versus combinational multiply in inspected RTL. The benchmark programs may also rely on Python-specific semantics such as arbitrary-width values before `c_int32` arithmetic.

## Scope recommendation for the heterogeneous study

The most isolatable DISCO-inspired block is an LCU-like loop controller. A narrow scalar LSU abstraction can also be compared if its request width, service rate, latency, and overlap are explicit; it must not be described as the DISCO LSU without SPM/VWR line behavior. MXCU should be omitted from the first experiment because its central role is VWR/SRF selection and transfer. Unused specialist positions in the proposed 2x7 layout should remain inert and contribute no compute capacity.

Treat the requested 2x7 arrangement as an experimental topology inspired by DISCO, not a reproduction of DISCO: the inspected DISCO compute array itself is 4x2. The primary heterogeneity comparison must be against an 8-compute-PE 2x7 reference; the 4x4 OE result is contextual because it changes compute capacity and mapping. Any cycle figures must name the Python timing model and its service assumptions.

## Benchmark availability

The current branch has MMUL CSV/configuration and runner files. The `gem`/`gemm` working directories contain only Python bytecode caches. The Git object history does contain earlier GEM/GEMM instruction CSVs and a runner on the separate SRF/Simple-Address experiment commits; those are historical reference material, not files in this branch, and their runner calls SRF-related simulator arguments absent from this base. The heterogeneous study should therefore construct and validate its own mapping in this branch, while recording that it is a new schedule and not the absent original GEM run. Do not report results from a mathematical reference calculation as CGRA measurements.
