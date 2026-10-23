#!/usr/bin/env python3
"""Run and validate the copied GEMM kernel with the repository CGRA model."""

import argparse
import csv
import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
INSTRUCTIONS = SCRIPT_DIR / "instructions.csv"
INSTRUCTION_FILES = {
    "original": INSTRUCTIONS,
    "srf_offsets": SCRIPT_DIR / "instructions_srf_offsets.csv",
    "srf_base": SCRIPT_DIR / "instructions_srf_base.csv",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cgra import N_COLS, N_ROWS, run  # noqa: E402
from kernels import kernel_add_memory_region, kernel_clear_memory  # noqa: E402


FIRST_ADDR = 20000
ALPHA = 125
BETA = 14
MAX_IT = 2_000_000


def configure_memory(kernel_name, a_data, b_data, c_data, rows_a, cols_a,
                     cols_b, version="original"):
    """Reproduce configMemory from benchmarks/gemm/test_gemm_meth.ipynb."""
    kernel_clear_memory(kernel_name)

    n_cols_blocks_c = cols_b // N_ROWS
    n_rows_blocks_c = rows_a // N_ROWS
    first_addr_a = FIRST_ADDR
    first_addr_b = first_addr_a + rows_a * cols_a * 4
    first_addr_c = first_addr_b + cols_a * cols_b * 4

    original_config = [
        [first_addr_b, n_cols_blocks_c,
         first_addr_a + 2 * cols_a * 4, first_addr_c + 3 * cols_b * 4,
         -4 * cols_b, BETA, cols_a],
        [first_addr_c + 4, first_addr_b + 4, cols_a,
         first_addr_a + 3 * cols_a * 4, cols_a, -4 * cols_b, BETA],
        [first_addr_a, first_addr_c + 2 * 4 + cols_b * 4,
         first_addr_b + 2 * 4, n_cols_blocks_c, ALPHA, cols_a, -4 * cols_b],
        [n_rows_blocks_c, first_addr_a + cols_a * 4,
         first_addr_c + 3 * 4 + 2 * cols_b * 4,
         first_addr_b + 3 * 4, ALPHA, cols_a, -4 * cols_b],
    ]

    srf_init = None
    if version == "original":
        config_by_col = original_config
    else:
        # PC0 obtains the first four per-RC R2 values either from memory
        # (Offsets) or directly from SRF0-SRF3 (Base). Alpha and beta are
        # supplied by SRF in both optimized variants, so omit them from the
        # scalar configuration stream.
        omitted_constants = ({5}, {6}, {4}, {4})
        filtered_config = [
            [value for index, value in enumerate(config)
             if index not in omitted_constants[col]]
            for col, config in enumerate(original_config)
        ]
        config_by_col = filtered_config
        if version == "srf_offsets":
            shared = {
                "SRF0": ALPHA, "SRF1": BETA,
                "SRF2": 16 - 4 * cols_a * cols_b,
                "SRF3": -4 * cols_a,
                "SRF4": 16 * cols_b,
                "SRF5": -4 * rows_a,
                "SRF6": 16 * cols_a,
                "SRF7": -4 * cols_b,
            }
            srf_init = [shared.copy() for _ in range(N_COLS)]
        else:
            srf_init = []
            for config in original_config:
                bank = {f"SRF{row}": config[row] for row in range(4)}
                bank.update({
                    "SRF4": ALPHA, "SRF5": BETA,
                    "SRF6": -4, "SRF7": 16,
                })
                srf_init.append(bank)
            config_by_col = [config[4:] for config in filtered_config]

    load_addrs = []
    address = 0
    for config in config_by_col:
        load_addrs.append(address)
        kernel_add_memory_region(kernel_name, address, config)
        address += len(config) * 4

    kernel_add_memory_region(kernel_name, first_addr_a, a_data)
    kernel_add_memory_region(kernel_name, first_addr_b, b_data)
    kernel_add_memory_region(kernel_name, first_addr_c, c_data)
    return load_addrs, first_addr_c, srf_init


def gemm_reference(a_data, b_data, c_data, rows_a, cols_a, cols_b):
    """Compute alpha * A*B + beta*C with the simulator's signed int32 result."""
    result = []
    for row in range(rows_a):
        for col in range(cols_b):
            product_sum = sum(
                a_data[row * cols_a + k] * b_data[k * cols_b + col]
                for k in range(cols_a)
            )
            value = ALPHA * product_sum + BETA * c_data[row * cols_b + col]
            result.append(((value + (1 << 31)) % (1 << 32)) - (1 << 31))
    return result


def read_result(path, first_addr_c, rows_a, cols_b):
    result = [0] * (rows_a * cols_b)
    written = set()
    with path.open(newline="") as memory_file:
        for row in csv.reader(memory_file):
            if len(row) < 2:
                continue
            try:
                address, value = int(row[0]), int(row[1])
            except ValueError:
                continue
            if first_addr_c <= address < first_addr_c + rows_a * cols_b * 4:
                index = (address - first_addr_c) // 4
                result[index] = value
                written.add(index)
    return result, len(written)


def execute(rows_a, cols_a, cols_b, version="original"):
    if N_ROWS != 4 or N_COLS != 4:
        raise RuntimeError(
            f"This kernel's notebook configuration requires a 4x4 CGRA; "
            f"cgra.py is configured as {N_ROWS}x{N_COLS}."
        )
    if min(rows_a, cols_a, cols_b) <= 0:
        raise ValueError("All GEMM dimensions must be positive.")
    if rows_a % N_ROWS or cols_b % N_COLS:
        raise ValueError(
            "The kernel computes complete 4x4 output tiles. rows-a and cols-b "
            "must be multiples of 4; cols-a may be any positive value."
        )
    instruction_file = INSTRUCTION_FILES[version]
    if not instruction_file.is_file():
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")

    # Match the deterministic initialization in the source notebook.
    a_data = list(range(rows_a * cols_a))
    b_data = [value + 100 for value in range(cols_a * cols_b)]
    c_data = [value + 200 for value in range(rows_a * cols_b)]
    expected = gemm_reference(a_data, b_data, c_data, rows_a, cols_a, cols_b)

    with tempfile.TemporaryDirectory(prefix="cgra-gemm-") as temporary_dir:
        kernel_name = "gemm_kernel"
        kernel_dir = Path(temporary_dir) / kernel_name
        kernel_dir.mkdir()
        shutil.copy2(instruction_file, kernel_dir / "instructions.csv")

        previous_cwd = Path.cwd()
        os.chdir(temporary_dir)
        try:
            load_addrs, first_addr_c, srf_init = configure_memory(
                kernel_name, a_data, b_data, c_data, rows_a, cols_a, cols_b,
                version=version,
            )
            simulation_output = io.StringIO()
            with redirect_stdout(simulation_output):
                cgra = run(
                    kernel_name,
                    pr=["ROUT", "INST"],
                    load_addrs=load_addrs,
                    limit=MAX_IT,
                    printVal=0,
                    srf_init=srf_init,
                    return_cgra=True,
                )
            output_path = kernel_dir / "memory_out.csv"
            actual, written = read_result(
                output_path, first_addr_c, rows_a, cols_b
            )
        finally:
            os.chdir(previous_cwd)

    if cgra is None or not cgra.exit:
        raise RuntimeError(
            "The CGRA did not reach EXIT. Simulator output was:\n"
            + simulation_output.getvalue()
        )
    cycles = len(cgra.instrs) * N_COLS + cgra.aproxcycles
    correct = written == rows_a * cols_b and actual == expected
    return correct, cycles, written, rows_a * cols_b


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and validate the 4x4-tiled GEMM CGRA kernel."
    )
    parser.add_argument(
        "--size", type=int, default=16,
        help="Use a square GEMM of this size for A, B, and C (default: 16).",
    )
    parser.add_argument(
        "--version", choices=tuple(INSTRUCTION_FILES), default="original",
        help="Instruction variant to execute.",
    )
    parser.add_argument("--rows-a", type=int, help="Rows in A and C.")
    parser.add_argument("--inner", type=int, help="Columns in A / rows in B.")
    parser.add_argument("--cols-b", type=int, help="Columns in B and C.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows_a = args.rows_a if args.rows_a is not None else args.size
    cols_a = args.inner if args.inner is not None else args.size
    cols_b = args.cols_b if args.cols_b is not None else args.size

    print(f"Instructions: {INSTRUCTION_FILES[args.version].name}")
    print(f"CGRA: {N_ROWS}x{N_COLS}; GEMM: {rows_a}x{cols_a}x{cols_b}")
    print(f"Operation: C = {ALPHA} * (A * B) + {BETA} * C")
    correct, cycles, written, total = execute(
        rows_a, cols_a, cols_b, version=args.version
    )
    print(f"Output values written by CGRA: {written}/{total}")
    print(f"Total simulated cycles: {cycles}")
    print(f"Validation: {'PASS' if correct else 'FAIL'}")
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
