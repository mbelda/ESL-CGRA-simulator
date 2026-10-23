#!/usr/bin/env python3
"""Run MMUL/GEMM on the original CGRA and row-LCU OpenEdge experiments.

The new mappings retain indirect loads/stores (LWI/SWI). Each output tile is
configured with its PE-local A/B/C pointers; the inner K loop uses each LCU's
four-register ALU and conditional branch. No direct stream pointers are used.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cgra
from cgra import CGRA, HeterogeneousCGRA, HeterogeneousPC


def input_matrices(n, seed=12345):
    rng = random.Random(seed)
    return ([rng.randint(-10, 10) for _ in range(n * n)],
            [rng.randint(-10, 10) for _ in range(n * n)],
            [rng.randint(-10, 10) for _ in range(n * n)])


def reference(a, b, c, n, gemm=False):
    out = []
    for i in range(n):
        for j in range(n):
            dot = sum(a[i * n + k] * b[k * n + j] for k in range(n))
            value = 125 * dot + 14 * c[i * n + j] if gemm else dot
            out.append(cgra.c_int32(value).value)
    return out


def tile_plan(n, rows):
    for row0 in range(0, n, rows):
        active_rows = min(rows, n - row0)
        for col0 in range(0, n, 4):
            yield row0, col0, active_rows


def pc(rows, lcu_instruction, pe_ops=None):
    if pe_ops is None:
        pe_ops = [["NOP"] * 4 for _ in range(rows)]
    return HeterogeneousPC(pe_ops, [lcu_instruction] * rows)


def build_lcu_program(n, rows, gemm, first_a, first_b, first_c,
                      lcu_enabled=True):
    program = []
    for row0, col0, active_rows in tile_plan(n, rows):
        # Initialize pointers and accumulator in PE registers, while each row
        # LCU initializes its private loop counter and bound.
        ptr_a = [["NOP"] * 4 for _ in range(rows)]
        for r in range(active_rows):
            for col in range(4):
                ptr_a[r][col] = f"SADD R0, ZERO, {first_a + 4 * ((row0 + r) * n)}"
        program.append(pc(rows, "SADD R0, ZERO, ZERO" if lcu_enabled else None, ptr_a))

        ptr_b = [["NOP"] * 4 for _ in range(rows)]
        for r in range(active_rows):
            for col in range(4):
                ptr_b[r][col] = f"SADD R2, ZERO, {first_b + 4 * (col0 + col)}"
        program.append(pc(rows, f"SADD R1, ZERO, {n}" if lcu_enabled else None, ptr_b))

        init_acc = [["NOP"] * 4 for _ in range(rows)]
        for r in range(active_rows):
            init_acc[r] = ["SADD R3, ZERO, ZERO"] * 4
        program.append(pc(rows, None, init_acc))

        loop_start = len(program)
        for iteration in range(n if not lcu_enabled else 1):
            load_a = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                load_a[r] = ["LWI R1, R0"] * 4
            program.append(pc(rows, "SADD R0, R0, 1" if lcu_enabled else None, load_a))

            load_b = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                for col in range(4):
                    load_b[r][col] = "LWI ROUT, R2"
            program.append(pc(rows, None, load_b))

            multiply = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                multiply[r] = ["SMUL ROUT, R1, ROUT"] * 4
            program.append(pc(rows, None, multiply))

            accumulate = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                accumulate[r] = ["SADD R3, R3, ROUT"] * 4
            program.append(pc(rows, None, accumulate))

            advance_a = [["NOP"] * 4 for _ in range(rows)]
            advance_b = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                advance_a[r] = ["SADD R0, R0, 4"] * 4
                advance_b[r] = [f"SADD R2, R2, {4 * n}"] * 4
            program.append(pc(rows, None, advance_a))
            program.append(pc(rows, f"BNE R0, R1, {loop_start}" if lcu_enabled else None,
                              advance_b))

        if gemm:
            scale_ab = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                scale_ab[r] = ["SMUL R3, R3, 125"] * 4
            program.append(pc(rows, None, scale_ab))

        ptr_c = [["NOP"] * 4 for _ in range(rows)]
        for r in range(active_rows):
            for col in range(4):
                address = first_c + 4 * ((row0 + r) * n + col0 + col)
                ptr_c[r][col] = f"SADD R0, ZERO, {address}"
        program.append(pc(rows, None, ptr_c))

        if gemm:
            load_c = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                load_c[r] = ["LWI R1, R0"] * 4
            program.append(pc(rows, None, load_c))
            scale_c = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                scale_c[r] = ["SMUL R1, R1, 14"] * 4
            program.append(pc(rows, None, scale_c))
            add_c = [["NOP"] * 4 for _ in range(rows)]
            for r in range(active_rows):
                add_c[r] = ["SADD R3, R3, R1"] * 4
            program.append(pc(rows, None, add_c))

        store = [["NOP"] * 4 for _ in range(rows)]
        for r in range(active_rows):
            store[r] = ["SWI R3, R0"] * 4
        program.append(pc(rows, None, store))
    return program


def run_heterogeneous(n, architecture, gemm, seed=12345):
    profile = cgra.ARCHITECTURES[architecture]
    a, b, c = input_matrices(n, seed)
    expected = reference(a, b, c, n, gemm)
    first_a, first_b, first_c = 100_000, 200_000, 300_000
    memory = []
    memory.extend([[first_a + 4 * i, x] for i, x in enumerate(a)])
    memory.extend([[first_b + 4 * i, x] for i, x in enumerate(b)])
    memory.extend([[first_c + 4 * i, x] for i, x in enumerate(c)])
    program = build_lcu_program(n, profile.rows, gemm,
                                first_a, first_b, first_c,
                                lcu_enabled=profile.lcu_per_row)
    sim = HeterogeneousCGRA(profile, program, memory)
    stats = sim.run(limit=20_000_000)
    stats.pop("memory", None)
    output = [None] * (n * n)
    for address, value in sim.memory:
        if first_c <= address < first_c + 4 * n * n and (address - first_c) % 4 == 0:
            output[(address - first_c) // 4] = value
    stats.update({
        "benchmark": "GEMM" if gemm else "MMUL",
        "architecture": architecture,
        "size": n,
        "seed": seed,
        "correct": output == expected,
        "written": sum(x is not None for x in output),
        "total_outputs": n * n,
    })
    if not stats["correct"]:
        stats["first_mismatch"] = next(
            (i for i, (x, y) in enumerate(zip(output, expected)) if x != y), None
        )
        stats["actual_head"] = output[:16]
        stats["expected_head"] = expected[:16]
    return stats


def _add_region(memory, address, values):
    memory.extend([[address + 4 * i, value] for i, value in enumerate(values)])


def run_original(n, gemm, seed=12345):
    """Run the preserved 4x4 instruction mapping with the same random data."""
    a, b, c = input_matrices(n, seed)
    expected = reference(a, b, c, n, gemm)
    first_a = 20_000
    first_b = first_a + 4 * n * n
    first_c = first_b + 4 * n * n
    if gemm:
        beta, alpha = 14, 125
        configs = [
            [first_b, n // 4, first_a + 2 * n * 4, first_c + 3 * n * 4,
             -4 * n, beta, n],
            [first_c + 4, first_b + 4, n, first_a + 3 * n * 4,
             n, -4 * n, beta],
            [first_a, first_c + 2 * 4 + n * 4, first_b + 2 * 4,
             n // 4, alpha, n, -4 * n],
            [n // 4, first_a + n * 4, first_c + 3 * 4 + 2 * n * 4,
             first_b + 3 * 4, alpha, n, -4 * n],
        ]
        instruction_path = ROOT / "benchmarks/DISCOvsOE/gemm/instructions.csv"
    else:
        configs = [
            [first_b, n // 4, first_a + 2 * n * 4, first_c + 3 * n * 4,
             -4 * n, n],
            [first_c + 4, first_b + 4, n, first_a + 3 * n * 4,
             n, -4 * n],
            [first_a, first_c + 2 * 4 + n * 4, first_b + 2 * 4,
             n // 4, n, -4 * n],
            [n // 4, first_a + n * 4, first_c + 3 * 4 + 2 * n * 4,
             first_b + 3 * 4, n, -4 * n],
        ]
        instruction_path = ROOT / "benchmarks/DISCOvsOE/mmul/instructions.csv"
    memory, load_addrs = [], []
    config_address = 0
    for config in configs:
        load_addrs.append(config_address)
        _add_region(memory, config_address, config)
        config_address += 4 * len(config)
    _add_region(memory, first_a, a)
    _add_region(memory, first_b, b)
    _add_region(memory, first_c, c)
    with instruction_path.open(newline="") as handle:
        kernel = list(csv.reader(handle))
    cgra.PRINT_OUTS = 0
    sim = CGRA(kernel, memory, load_addrs, None)
    with contextlib.redirect_stdout(io.StringIO()):
        final_memory = sim.run(pr=[], limit=20_000_000)
    actual = [None] * (n * n)
    for address, value in final_memory:
        if first_c <= address < first_c + n * n * 4 and (address - first_c) % 4 == 0:
            actual[(address - first_c) // 4] = value
    return {
        "benchmark": "GEMM" if gemm else "MMUL",
        "architecture": "OE_4x4",
        "size": n,
        "seed": seed,
        "correct": actual == expected,
        "written": sum(x is not None for x in actual),
        "total_outputs": n * n,
        "execution_cycles": sim.aproxcycles,
        "configuration_cycles": len(sim.instrs) * cgra.N_COLS,
        "total_cycles": sim.aproxcycles + len(sim.instrs) * cgra.N_COLS,
        "memory_cycles": sim.aproxMemCycles,
        "arithmetic_cycles": sim.aproxArithCycles,
        "pcs_executed": sim.cycles,
    }


def run_case(benchmark, architecture, size, seed=12345):
    gemm = benchmark.lower() == "gemm"
    if architecture == "OE_4x4":
        return run_original(size, gemm, seed)
    return run_heterogeneous(size, architecture, gemm, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("mmul", "gemm"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--architecture", choices=tuple(cgra.ARCHITECTURES), required=True)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    result = run_case(args.benchmark, args.architecture, args.size, args.seed)
    print(json.dumps(result, indent=2))
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
