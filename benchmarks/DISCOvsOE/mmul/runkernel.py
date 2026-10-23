import sys
import argparse
import csv
from contextlib import redirect_stdout
import random

from cgra import *
from kernels import *

kernel_name = "benchmarks/DISCOvsOE/mmul"
version = ""
srf_init = None

# Global variables
CGRA_N_ROWS = 4
CGRA_N_COLS = 4
first_addr = 20000
DEBUG = False


def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])


def configMemory(A_data, B_data, C_data, rowsA, colsA, colsB):
    kernel_clear_memory(kernel_name, version=version)

    nItLoopColsA = colsA
    nColsBlocksC = int(colsB / CGRA_N_ROWS)
    nRowsBlocksC = int(rowsA / CGRA_N_ROWS)

    first_addr_A = first_addr
    first_addr_B = first_addr_A + rowsA * colsA * 4
    first_addr_C = first_addr_B + colsA * colsB * 4

    config_vals_col0 = [first_addr_B, nColsBlocksC,
                        first_addr_A + 2 * colsA * 4,
                        first_addr_C + 3 * colsB * 4,
                        -4 * colsB, colsA]

    config_vals_col1 = [first_addr_C + 4, first_addr_B + 4,
                        nItLoopColsA,
                        first_addr_A + 3 * colsA * 4,
                        colsA, -4 * colsB]

    config_vals_col2 = [first_addr_A,
                        first_addr_C + 2 * 4 + colsB * 4,
                        first_addr_B + 2 * 4,
                        nColsBlocksC, colsA, -4 * colsB]

    config_vals_col3 = [nRowsBlocksC,
                        first_addr_A + colsA * 4,
                        first_addr_C + 3 * 4 + 2 * colsB * 4,
                        first_addr_B + 3 * 4,
                        colsA, -4 * colsB]

    addr = 0
    for cfg in [config_vals_col0, config_vals_col1,
                config_vals_col2, config_vals_col3]:
        kernel_add_memory_region(kernel_name, addr, cfg, version=version)
        addr += len(cfg) * 4

    kernel_add_memory_region(kernel_name, first_addr_A, A_data, version=version)
    kernel_add_memory_region(kernel_name, first_addr_B, B_data, version=version)
    kernel_add_memory_region(kernel_name, first_addr_C, C_data, version=version)

    return [
        0,
        len(config_vals_col0) * 4,
        (len(config_vals_col0) + len(config_vals_col1)) * 4,
        (len(config_vals_col0) + len(config_vals_col1) + len(config_vals_col2)) * 4
    ]


def runKernel(load_addrs, max_it=1000):
    pr = []
    if DEBUG: 
        pr = ["ROUT", "INST"]
    return run(kernel_name,
        pr=pr,
        load_addrs=load_addrs,
        version=version,
        limit=max_it,
        srf_init=srf_init,
        printVal=1 if DEBUG else 0,
        return_cgra=True)


def getResult(first_addr_C, end_addr_C, rowsA, colsB):
    result = [0 for _ in range(rowsA * colsB)]
    with open(kernel_name + "/memory_out" + version + ".csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            try:
                addr = int(row[0])
                if first_addr_C <= addr < end_addr_C:
                    result[(addr - first_addr_C) // 4] = int(row[1])
            except ValueError:
                print("Error: Values in memory_out CSV file are not integers.")
    return result


def mmul_cpu(A_data, B_data, rowsA, colsA, colsB):
    out = [0 for _ in range(rowsA * colsB)]
    for r in range(rowsA):
        for c in range(colsB):
            s = 0
            for k in range(colsA):
                s += A_data[r * colsA + k] * B_data[k * colsB + c]
            out[r * colsB + c] = s
    return out


# ---------------------- MAIN ----------------------

parser = argparse.ArgumentParser()
parser.add_argument("rowsA", type=int)
parser.add_argument("colsA", type=int)
parser.add_argument("colsB", type=int)
parser.add_argument("--version", choices=("original", "srf_offsets", "srf_base"), default="original")
parser.add_argument("--seed", type=int, default=12345,
                    help="seed used to generate the same A/B/C matrices across versions")
args = parser.parse_args()

rowsA, colsA, colsB = args.rowsA, args.colsA, args.colsB
version = {"original": "", "srf_offsets": "_srf_offsets", "srf_base": "_srf_base"}[args.version]
logfile = f"out_mmul_{rowsA}x{colsA}x{colsB}.log"


def configMemory_srf_offsets(A_data, B_data, C_data, rowsA, colsA, colsB):
    """Configure matrix memory and dimension-derived address offsets in each SRF bank."""
    global srf_init
    addrs = configMemory(A_data, B_data, C_data, rowsA, colsA, colsB)
    common = {"SRF0": colsA, "SRF1": -4 * colsB,
              "SRF2": -4 * colsA * colsB + 16, "SRF3": -4 * colsA,
              "SRF4": 16 * colsB, "SRF5": -4 * rowsA,
              "SRF6": 16 * colsA}
    srf_init = [common.copy() for _ in range(CGRA_N_COLS)]
    return addrs


def configMemory_srf_base(A_data, B_data, C_data, rowsA, colsA, colsB):
    """Place each RC's initial R2 base/config value in its column SRF."""
    global srf_init
    kernel_clear_memory(kernel_name, version=version)
    n_cols_blocks = colsB // CGRA_N_ROWS
    n_rows_blocks = rowsA // CGRA_N_ROWS
    addr_a = first_addr
    addr_b = addr_a + rowsA * colsA * 4
    addr_c = addr_b + colsA * colsB * 4
    kernel_add_memory_region(kernel_name, addr_a, A_data, version=version)
    kernel_add_memory_region(kernel_name, addr_b, B_data, version=version)
    kernel_add_memory_region(kernel_name, addr_c, C_data, version=version)
    # These per-column, per-row values are the original R2 pointer/bound bases.
    # SRF0-SRF3 keep them; the A-row entry in each column is refreshed at the
    # row-band transition and then serves as the direct rewind point.
    row_values = [
        [addr_b, n_cols_blocks, addr_a + 2 * colsA * 4, addr_c + 3 * colsB * 4],
        [addr_c + 4, addr_b + 4, colsA, addr_a + 3 * colsA * 4],
        [addr_a, addr_c + 2 * 4 + colsB * 4, addr_b + 2 * 4, n_cols_blocks],
        [n_rows_blocks, addr_a + colsA * 4, addr_c + 3 * 4 + 2 * colsB * 4, addr_b + 3 * 4],
    ]
    srf_init = []
    for col in range(CGRA_N_COLS):
        bank = {f"SRF{row}": row_values[col][row] for row in range(CGRA_N_ROWS)}
        bank.update({"SRF4": colsA, "SRF5": -4 * colsB,
                     "SRF6": -4 * colsA * colsB})
        srf_init.append(bank)
    # All pointer/configuration state is initialized in SRF; no scalar config
    # loads occur before entering the kernel.
    return [0] * CGRA_N_COLS


if min(rowsA, colsA, colsB) <= 0 or any(
        dimension % CGRA_N_ROWS for dimension in (rowsA, colsA, colsB)):
    parser.error("rowsA, colsA and colsB must be positive multiples of 4")

with open(logfile, "w") as log, redirect_stdout(log):
    random.seed(args.seed)
    A_data = [random.randint(-10, 10) for _ in range(rowsA * colsA)]
    B_data = [random.randint(-10, 10) for _ in range(colsA * colsB)]
    C_data = [random.randint(-10, 10) for _ in range(rowsA * colsB)]

    if args.version == "srf_offsets":
        load_addrs = configMemory_srf_offsets(A_data, B_data, C_data,
                                              rowsA, colsA, colsB)
    elif args.version == "srf_base":
        load_addrs = configMemory_srf_base(A_data, B_data, C_data,
                                           rowsA, colsA, colsB)
    else:
        load_addrs = configMemory(A_data, B_data, C_data,
                                  rowsA, colsA, colsB)

    cgra = runKernel(load_addrs, max_it=2_000_000)
    if cgra is None or not cgra.exit:
        raise RuntimeError("CGRA simulation did not reach EXIT before the execution limit")

    first_addr_C = first_addr + rowsA * colsA * 4 + colsA * colsB * 4
    result = getResult(first_addr_C,
                       first_addr_C + rowsA * colsB * 4,
                       rowsA, colsB)

    expected = mmul_cpu(A_data, B_data, rowsA, colsA, colsB)

    errors = sum(1 for i in range(len(expected))
                 if expected[i] != result[i])

# --------- OUTPUT MINIMAL POR TERMINAL ---------

with open(logfile, "r") as log:
    lines = log.readlines()
    for l in lines[-5:]:
        print(l.rstrip())

if errors > 0:
    print(f"Err: {errors}")
    raise SystemExit(1)
else:
    print("OK")
print(f"CYCLES total={len(cgra.instrs) * CGRA_N_COLS + cgra.aproxcycles} "
      f"config={len(cgra.instrs) * CGRA_N_COLS} execution={cgra.aproxcycles} "
      f"memory={cgra.aproxMemCycles} arithmetic={cgra.aproxArithCycles}")
phase_cycles = {"prep": 0, "compute": 0, "load_store": 0, "address_update": 0, "exit": 0}
if args.version == "srf_offsets":
    # The fused instruction stream has 17 PCs: setup 0-1, compute 2-7,
    # load/store 8-11, address updates 12-15, and EXIT at 16.
    for pc, cycles in cgra.cycles_by_pc.items():
        if pc <= 1:
            phase_cycles["prep"] += cycles
        elif pc <= 7:
            phase_cycles["compute"] += cycles
        elif pc <= 11:
            phase_cycles["load_store"] += cycles
        elif pc <= 15:
            phase_cycles["address_update"] += cycles
        else:
            phase_cycles["exit"] += cycles
else:
    if args.version == "original":
        orig_pc = {pc: pc for pc in cgra.cycles_by_pc}
    else:
        orig_pc = {pc: (pc if pc < 2 else pc + 1 if pc <= 12 else pc + 2)
                   for pc in cgra.cycles_by_pc}
    for pc, cycles in cgra.cycles_by_pc.items():
        source_pc = orig_pc[pc]
        if source_pc <= 2:
            phase_cycles["prep"] += cycles
        elif source_pc <= 8:
            phase_cycles["compute"] += cycles
        elif source_pc <= 12:
            phase_cycles["load_store"] += cycles
        elif source_pc <= 19:
            phase_cycles["address_update"] += cycles
        else:
            phase_cycles["exit"] += cycles
print("PHASE_CYCLES " + " ".join(f"{key}={value}" for key, value in phase_cycles.items()))
