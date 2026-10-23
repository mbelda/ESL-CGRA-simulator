#!/usr/bin/env python3
"""Isolate loop-control offload on the unchanged manual 4x4 kernels.

The PE instruction stream is kept at its original 4x4 geometry.  In the LCU
cases, only each branch opcode is removed from its PE slot; the branch is
evaluated concurrently from that PE's existing loop registers.  A control
sideband reproduces the branch instruction's Rout value so that PE dataflow
remains bit-for-bit equivalent.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cgra
from cgra import CGRA, LCU

BENCHMARK_FILES = {
    "MMUL": ROOT / "benchmarks/DISCOvsOE/mmul/instructions.csv",
    "GEM": ROOT / "benchmarks/DISCOvsOE/gemm/instructions.csv",
}
SIZES = (16, 32, 64, 128)
MODES = ("OE_4x4", "OE_4x4_1LCU", "OE_4x5_4LCU")


@dataclass(frozen=True)
class Branch:
    pc: int
    row: int
    col: int
    op: str
    lhs: str
    rhs: str
    target: str


def input_matrices(n: int, seed: int = 12345):
    rng = random.Random(seed)
    return ([rng.randint(-10, 10) for _ in range(n * n)],
            [rng.randint(-10, 10) for _ in range(n * n)],
            [rng.randint(-10, 10) for _ in range(n * n)])


def reference(a, b, c, n, gem=False):
    out = []
    for i in range(n):
        for j in range(n):
            dot = sum(a[i * n + k] * b[k * n + j] for k in range(n))
            value = 125 * dot + 14 * c[i * n + j] if gem else dot
            out.append(cgra.c_int32(value).value)
    return out


def add_region(memory, address, values):
    memory.extend([[address + 4 * i, value] for i, value in enumerate(values)])


def read_kernel(benchmark):
    with BENCHMARK_FILES[benchmark].open(newline="") as handle:
        return list(csv.reader(handle))


def extract_branches(instructions):
    branches = []
    for pc, instruction in enumerate(instructions):
        for row in range(cgra.N_ROWS):
            for col in range(cgra.N_COLS):
                words = instruction.ops[row][col].replace(",", " ").split()
                if words and words[0].upper() in ("BEQ", "BNE", "BLT", "BGE"):
                    if len(words) != 4:
                        raise ValueError(f"Malformed branch at PC {pc}, RC({row},{col})")
                    branches.append(Branch(pc, row, col, words[0].upper(),
                                           words[1], words[2], words[3]))
    return branches


def compare(op, lhs, rhs):
    return {"BEQ": lhs == rhs, "BNE": lhs != rhs,
            "BLT": lhs < rhs, "BGE": lhs >= rhs}[op]


class ControlCompareCGRA(CGRA):
    """Original 4x4 PE model with an optional parallel LCU branch path."""

    def __init__(self, kernel, memory, read_addrs, mode, trace=False):
        super().__init__(kernel, memory, read_addrs, None)
        self.mode = mode
        self.trace_enabled = trace
        self.original_branches = extract_branches(self.instrs)
        self.branches_by_pc = {branch.pc: branch for branch in self.original_branches}
        if len(self.branches_by_pc) != len(self.original_branches):
            raise ValueError("More than one branch in a PC is not supported by this kernel")
        self.lcus = ([LCU("global")] if mode == "OE_4x4_1LCU" else
                     [LCU(row) for row in range(cgra.N_ROWS)]
                     if mode == "OE_4x5_4LCU" else [])
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if self.lcus:
            for branch in self.original_branches:
                # Keep the PC and all other RC instructions. A branch-only
                # datapath slot becomes idle while its control moves to LCU.
                self.instrs[branch.pc].ops[branch.row][branch.col] = "NOP"

        # Instruction text is static across a run. Cache parsed PE operations
        # so large matrices do not repeatedly tokenize each PC's 16 slots.
        self.data_ops_by_pc = [self._parse_data_pc_ops(pc)
                               for pc in range(len(self.instrs))]
        self.pc_latency = []
        for pc in self.instrs:
            memory_ops = multiply_ops = 0
            for row in pc.ops:
                for text in row:
                    words = text.replace(",", " ").split()
                    if words and words[0].upper() in ("LWD", "LWI", "SWD", "SWI"):
                        memory_ops += 1
                    if words and words[0].upper() == "SMUL":
                        multiply_ops += 1
            latency = max(1 + memory_ops if memory_ops else 1,
                          3 if multiply_ops else 1)
            self.pc_latency.append((memory_ops, latency))

        self.pc_visits = Counter()
        self.op_visits = Counter()
        self.branch_visits = Counter()
        self.branch_taken = Counter()
        self.lcu_issue_cycles = 0
        self.lcu_operation_count = 0
        self.lcu_branch_count = 0
        self.lcu_sync_stalls = 0
        self.lcu_operations_by_unit = Counter()
        self.control_digest = hashlib.sha256()
        self.trace_digest = hashlib.sha256()

    def _parse_data_pc_ops(self, pc):
        ops = []
        for row in range(cgra.N_ROWS):
            for col in range(cgra.N_COLS):
                words = self.instrs[pc].ops[row][col].replace(",", " ").split()
                if not words:
                    continue
                opcode = words[0].upper()
                if opcode in ("NOP", "EXIT", "BEQ", "BNE", "BLT", "BGE", "JUMP"):
                    continue
                ops.append((row, col, opcode, words))
        return ops

    def _data_pc_ops(self, pc):
        return self.data_ops_by_pc[pc]

    def maxCyclesThisPC(self, pc_index=None):
        if pc_index is None:
            pc_index = self.instr2exec
        memory_ops, latency = self.pc_latency[pc_index]
        if memory_ops:
            self.aproxMemCycles += 1 + memory_ops
        else:
            self.aproxArithCycles += latency
        return latency

    def _trace_inputs(self, pc):
        """Hash issued data operations and their pre-issue operand values."""
        if not self.trace_enabled:
            return
        for row, col, opcode, words in self._data_pc_ops(pc):
            cell = self.cells[row][col]
            if opcode in ("LWD", "LWI"):
                source_tokens = words[2:3]
            elif opcode in ("SWD", "SWI"):
                source_tokens = words[1:3]
            else:
                source_tokens = words[2:4]
            values = tuple(cell.fetch_val(token) for token in source_tokens)
            item = (pc, row, col, opcode, tuple(source_tokens), values)
            self.trace_digest.update(repr(item).encode("ascii"))

    def _trace_state(self, pc):
        if not self.trace_enabled:
            return
        state = []
        for row in self.cells:
            for cell in row:
                state.append((tuple(cell.regs[name] for name in ("R0", "R1", "R2", "R3")),
                              cell.out, cell.old_out, cell.flags["zero"],
                              cell.flags["sign"], cell.flags["exit"]))
        self.trace_digest.update(repr((pc, tuple(state))).encode("ascii"))
        # Include memory checkpoints at PCs that store. This keeps the
        # trace bounded while checking every output write in the small case.
        if any(opcode in ("SWD", "SWI") for _, _, opcode, _ in self._data_pc_ops(pc)):
            self.trace_digest.update(repr(tuple(tuple(item) for item in self.memory)).encode("ascii"))

    def step(self, prs="ROUT"):
        pc = self.instr2exec
        self.pc_visits[pc] += 1
        self._trace_inputs(pc)
        for _, _, opcode, _ in self._data_pc_ops(pc):
            self.op_visits[opcode] += 1

        branch = self.branches_by_pc.get(pc)
        branch_values = None
        lcu_taken = None
        if branch is not None:
            source = self.cells[branch.row][branch.col]
            lhs = source.fetch_val(branch.lhs)
            rhs = source.fetch_val(branch.rhs)
            target = source.fetch_val(branch.target)
            taken = compare(branch.op, lhs, rhs)
            branch_values = (lhs, rhs, target, taken)
            self.branch_visits[pc] += 1
            if taken:
                self.branch_taken[pc] += 1
            if self.lcus:
                # The global unit evaluates once. The four-row design
                # broadcasts one predicate to every LCU; they evaluate it in
                # parallel and must agree on the same global PC target.
                active_lcus = self.lcus
                lcu_targets = []
                for lcu in active_lcus:
                    lcu.regs["R0"] = lhs
                    lcu.regs["R1"] = rhs
                    self.lcu_operation_count += 1
                    self.lcu_branch_count += 1
                    self.lcu_operations_by_unit[str(lcu.row)] += 1
                    lcu_targets.append(lcu.execute(
                        f"{branch.op} R0 R1 {branch.target}", pc
                    ))
                expected_target = target if taken else None
                if any(value != expected_target for value in lcu_targets):
                    raise RuntimeError(f"LCU disagreement at PC {pc}: {lcu_targets}")
                lcu_taken = expected_target
                self.lcu_issue_cycles += 1

        done = super().step(prs)

        if branch is not None:
            lhs, rhs, target, taken = branch_values
            if self.lcus:
                # The branch PE normally writes its target operand to Rout,
                # regardless of whether the condition is taken. Reproduce
                # that value over the explicit control/data sideband.
                self.cells[branch.row][branch.col].out = target
                if lcu_taken is not None:
                    self.instr2exec = target
            else:
                pe_taken = bool(self.cells[branch.row][branch.col].flags["branch"])
                if pe_taken != taken:
                    raise RuntimeError(f"PE branch predicate mismatch at PC {pc}")
            self.control_digest.update(
                repr((pc, branch.row, branch.col, branch.op,
                      lhs, rhs, target, taken)).encode("ascii")
            )

        self._trace_state(pc)
        return done


def run_case(benchmark, size, mode, seed=12345, trace=False):
    benchmark = benchmark.upper()
    if size % 4:
        raise ValueError("The original bitstreams require matrix dimensions divisible by 4")
    a, b, c = input_matrices(size, seed)
    expected = reference(a, b, c, size, benchmark == "GEM")
    first_a = 20_000
    first_b = first_a + 4 * size * size
    first_c = first_b + 4 * size * size
    if benchmark == "GEM":
        alpha, beta = 125, 14
        configs = [
            [first_b, size // 4, first_a + 2 * size * 4,
             first_c + 3 * size * 4, -4 * size, beta, size],
            [first_c + 4, first_b + 4, size, first_a + 3 * size * 4,
             size, -4 * size, beta],
            [first_a, first_c + 2 * 4 + size * 4, first_b + 2 * 4,
             size // 4, alpha, size, -4 * size],
            [size // 4, first_a + size * 4,
             first_c + 3 * 4 + 2 * size * 4, first_b + 3 * 4,
             alpha, size, -4 * size],
        ]
    elif benchmark == "MMUL":
        configs = [
            [first_b, size // 4, first_a + 2 * size * 4,
             first_c + 3 * size * 4, -4 * size, size],
            [first_c + 4, first_b + 4, size, first_a + 3 * size * 4,
             size, -4 * size],
            [first_a, first_c + 2 * 4 + size * 4, first_b + 2 * 4,
             size // 4, size, -4 * size],
            [size // 4, first_a + size * 4,
             first_c + 3 * 4 + 2 * size * 4, first_b + 3 * 4,
             size, -4 * size],
        ]
    else:
        raise ValueError(f"Unknown benchmark {benchmark!r}")

    memory, load_addrs = [], []
    config_address = 0
    for config in configs:
        load_addrs.append(config_address)
        add_region(memory, config_address, config)
        config_address += 4 * len(config)
    add_region(memory, first_a, a)
    add_region(memory, first_b, b)
    add_region(memory, first_c, c)

    kernel = read_kernel(benchmark)
    cgra.PRINT_OUTS = 0
    sim = ControlCompareCGRA(kernel, memory, load_addrs, mode, trace=trace)
    with contextlib.redirect_stdout(io.StringIO()):
        final_memory = sim.run(pr=[], limit=20_000_000)

    actual = [None] * (size * size)
    for address, value in final_memory:
        if first_c <= address < first_c + 4 * size * size and (address - first_c) % 4 == 0:
            actual[(address - first_c) // 4] = value
    pe_config_cycles = len(sim.instrs) * cgra.N_COLS
    # The model loads one LCU instruction context word per distinct branch PC.
    # Four row LCUs are configured in parallel, so this is not multiplied by 4.
    lcu_config_cycles = len(sim.branches_by_pc) if sim.lcus else 0
    result = {
        "benchmark": benchmark,
        "size": size,
        "architecture": mode,
        "seed": seed,
        "correct": actual == expected,
        "written": sum(value is not None for value in actual),
        "total_outputs": size * size,
        "execution_cycles": sim.aproxcycles,
        "pe_configuration_cycles": pe_config_cycles,
        "lcu_configuration_cycles": lcu_config_cycles,
        "configuration_cycles": pe_config_cycles + lcu_config_cycles,
        "total_cycles": sim.aproxcycles + pe_config_cycles + lcu_config_cycles,
        "pcs_executed": sim.cycles,
        "pc_visits": {str(pc): count for pc, count in sorted(sim.pc_visits.items())},
        "dynamic_pe_operations": dict(sorted(sim.op_visits.items())),
        "control_pcs_static": sorted(sim.branches_by_pc),
        "control_branch_events": sum(sim.branch_visits.values()),
        "control_taken_branches": sum(sim.branch_taken.values()),
        "branch_events_by_pc": {str(pc): count for pc, count in sorted(sim.branch_visits.items())},
        "taken_branches_by_pc": {str(pc): count for pc, count in sorted(sim.branch_taken.items())},
        "control_trace_sha256": sim.control_digest.hexdigest(),
        "lcu_units": len(sim.lcus),
        "lcu_operations": sim.lcu_operation_count,
        "lcu_branch_operations": sim.lcu_branch_count,
        "lcu_operations_by_unit": dict(sorted(sim.lcu_operations_by_unit.items())),
        "lcu_issue_cycles": sim.lcu_issue_cycles,
        "lcu_sync_stalls": sim.lcu_sync_stalls,
    }
    if trace:
        result["data_trace_sha256"] = sim.trace_digest.hexdigest()
    if not result["correct"]:
        result["first_mismatch"] = next(
            (i for i, (got, want) in enumerate(zip(actual, expected)) if got != want), None
        )
        result["actual_head"] = actual[:16]
        result["expected_head"] = expected[:16]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run both kernels at 16, 32, 64, 128")
    parser.add_argument("--benchmark", choices=("mmul", "gem"))
    parser.add_argument("--size", type=int)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--trace", action="store_true", help="Record a detailed dynamic dataflow trace")
    parser.add_argument("--output", type=Path, default=HERE / "same_mesh_results.json")
    args = parser.parse_args()

    if args.all:
        results = []
        # First compare detailed traces at a single 4x4 output tile.
        trace_results = []
        for benchmark in ("MMUL", "GEM"):
            small = [run_case(benchmark, 4, mode, args.seed, trace=True) for mode in MODES]
            if len({row["data_trace_sha256"] for row in small}) != 1:
                raise RuntimeError(f"Dataflow trace mismatch for {benchmark} at N=4")
            if len({row["control_trace_sha256"] for row in small}) != 1:
                raise RuntimeError(f"Control trace mismatch for {benchmark} at N=4")
            trace_results.extend(small)
            print(f"TRACE PASS {benchmark}: N=4, all PE data and control traces identical", flush=True)

        for benchmark in ("MMUL", "GEM"):
            for size in SIZES:
                cases = [run_case(benchmark, size, mode, args.seed) for mode in MODES]
                if not all(row["correct"] for row in cases):
                    raise RuntimeError(f"Incorrect output: {cases}")
                if len({row["control_trace_sha256"] for row in cases}) != 1:
                    raise RuntimeError(f"Control trace mismatch: {benchmark} N={size}")
                if len({row["execution_cycles"] for row in cases}) != 1:
                    raise RuntimeError(f"Execution cycle mismatch: {benchmark} N={size}")
                if len({json.dumps(row["pc_visits"], sort_keys=True) for row in cases}) != 1:
                    raise RuntimeError(f"PC traversal mismatch: {benchmark} N={size}")
                if len({json.dumps(row["dynamic_pe_operations"], sort_keys=True)
                        for row in cases}) != 1:
                    raise RuntimeError(f"PE operation count mismatch: {benchmark} N={size}")
                results.extend(cases)
                for row in cases:
                    print(f"PASS {benchmark} N={size} {row['architecture']}: "
                          f"exec={row['execution_cycles']}, total={row['total_cycles']}, "
                          f"control={row['control_branch_events']} "
                          f"({row['control_taken_branches']} taken)", flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        trace_path = args.output.with_name("same_mesh_trace_validation.json")
        trace_path.write_text(json.dumps(trace_results, indent=2) + "\n")
        csv_path = args.output.with_suffix(".csv")
        fields = ["benchmark", "size", "architecture", "correct", "written",
                  "total_outputs", "execution_cycles", "pe_configuration_cycles",
                  "lcu_configuration_cycles", "configuration_cycles", "total_cycles",
                  "pcs_executed", "control_pcs_static", "control_branch_events",
                  "control_taken_branches", "branch_events_by_pc", "taken_branches_by_pc",
                  "lcu_units", "lcu_operations", "lcu_branch_operations",
                  "lcu_operations_by_unit", "lcu_issue_cycles", "lcu_sync_stalls",
                  "control_trace_sha256"]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow({key: json.dumps(row[key], sort_keys=True)
                                 if isinstance(row.get(key), (dict, list)) else row.get(key)
                                 for key in fields})
        print(f"Wrote {len(results)} runs to {csv_path} and {args.output}")
        return 0

    if not args.benchmark or args.size is None or not args.mode:
        parser.error("select --all or provide --benchmark, --size, and --mode")
    result = run_case(args.benchmark, args.size, args.mode, args.seed, args.trace)
    print(json.dumps(result, indent=2))
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
