import sys
import argparse
import csv
from contextlib import redirect_stdout
import random

from cgra import *
from kernels import *

kernel_name = "benchmarks/disco-vs-oe/transpose-scale"
version = ""

# Global variables
CGRA_N_ROWS = 4
CGRA_N_COLS = 4
first_addr = 20000
DEBUG = False

def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])


def configMemory(A_data, rowsA, colsA, scale_f):
    kernel_clear_memory(kernel_name, version=version)

    # ------------------------------------------
    # CONFIGURATION
    # ------------------------------------------
    # nItLoopBlocksCols     A[0][1]               A[0][2]     A[0][3]
    # A[1][0]               nItLoopBlocksRows     A[1][2]     A[1][3]
    # A[2][0]               A[2][1]               A[2][2]     A[2][3]
    # A[3][0]               A[3][1]               A[3][2]     A[3][3]
    # ------------------------------------------
    # B[0][0]               B[0][1]               B[0][2]     B[0][3]
    # B[1][0]               B[1][1]               B[1][2]     B[1][3]
    # B[2][0]               B[2][1]               B[2][2]     B[2][3]
    # B[3][0]               B[3][1]               B[3][2]     B[3][3]
    # -------------------------------------------
    # scale_f               scale_f               scale_f     scale_f
    # scale_f               scale_f               scale_f     scale_f
    # scale_f               scale_f               scale_f     scale_f
    # scale_f               scale_f               scale_f     scale_f
    # -------------------------------------------
    # -                     colsIn                colsOut     16*(nBlocksColsi + 1)
    # 16*(nBlocksColsi + 1) -                     colsIn      colsOut
    # colsOut               16*(nBlocksColsi + 1) -           colsIn
    # colsIn                colsOut               16*(nBlocksColsi + 1) -
    # -------------------------------------------

    colsB = rowsA # B = At
    nItLoopBlocksCols = colsA // CGRA_N_COLS
    nItLoopBlocksRows = rowsA // CGRA_N_ROWS

    first_addr_A = first_addr
    first_addr_B = first_addr_A + rowsA * colsA * 4

    config_vals_col0 = [
        nItLoopBlocksCols,
        first_addr_A + (1*colsA + 0)*4,
        first_addr_A + (2*colsA + 0)*4,
        first_addr_A + (3*colsA + 0)*4,

        first_addr_B + (0*colsB + 0)*4,
        first_addr_B + (1*colsB + 0)*4,
        first_addr_B + (2*colsB + 0)*4,
        first_addr_B + (3*colsB + 0)*4,

        scale_f, scale_f, scale_f, scale_f,

        16*(nItLoopBlocksCols + 1),
        colsB,
        colsA
    ]


    config_vals_col1 = [
        first_addr_A + (0*colsA + 1)*4,
        nItLoopBlocksRows,
        first_addr_A + (2*colsA + 1)*4,
        first_addr_A + (3*colsA + 1)*4,

        first_addr_B + (0*colsB + 1)*4,
        first_addr_B + (1*colsB + 1)*4,
        first_addr_B + (2*colsB + 1)*4,
        first_addr_B + (3*colsB + 1)*4,

        scale_f, scale_f, scale_f, scale_f,

        colsA,
        16*(nItLoopBlocksCols + 1),
        colsB
    ]


    config_vals_col2 = [
        first_addr_A + (0*colsA + 2)*4,
        first_addr_A + (1*colsA + 2)*4,
        first_addr_A + (2*colsA + 2)*4,
        first_addr_A + (3*colsA + 2)*4,

        first_addr_B + (0*colsB + 2)*4,
        first_addr_B + (1*colsB + 2)*4,
        first_addr_B + (2*colsB + 2)*4,
        first_addr_B + (3*colsB + 2)*4,

        scale_f, scale_f, scale_f, scale_f,

        colsB,
        colsA,
        16*(nItLoopBlocksCols + 1)
    ]

    config_vals_col3 = [
        first_addr_A + (0*colsA + 3)*4,
        first_addr_A + (1*colsA + 3)*4,
        first_addr_A + (2*colsA + 3)*4,
        first_addr_A + (3*colsA + 3)*4,

        first_addr_B + (0*colsB + 3)*4,
        first_addr_B + (1*colsB + 3)*4,
        first_addr_B + (2*colsB + 3)*4,
        first_addr_B + (3*colsB + 3)*4,

        scale_f, scale_f, scale_f, scale_f,

        16*(nItLoopBlocksCols + 1),
        colsB,
        colsA
    ]



    addr = 0
    for cfg in [config_vals_col0, config_vals_col1,
                config_vals_col2, config_vals_col3]:
        kernel_add_memory_region(kernel_name, addr, cfg, version=version)
        addr += len(cfg) * 4

    kernel_add_memory_region(kernel_name, first_addr_A, A_data, version=version)

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
    run(kernel_name,
        pr=pr,
        load_addrs=load_addrs,
        version=version,
        limit=max_it)


def getResult(first_addr_C, end_addr_C, length):
    result = [0 for _ in range(length)]
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


def transpose_and_scale_cpu(A_data, rowsA, colsA, scale_f):
    out = [0 for _ in range(rowsA * colsA)]
    for i in range(rowsA):
        for j in range(colsA):
            out[j*rowsA + i] = A_data[i*colsA + j] >> scale_f
    return out


# ---------------------- MAIN ----------------------

parser = argparse.ArgumentParser()
parser.add_argument("rowsA", type=int)
parser.add_argument("colsA", type=int)
parser.add_argument("scale_f", type=int)
args = parser.parse_args()

rowsA, colsA, scale_f = args.rowsA, args.colsA, args.scale_f
logfile = f"out_tras_scale_{rowsA}x{colsA}_sf{scale_f}.log"

with open(logfile, "w") as log, redirect_stdout(log):

    A_data = [random.randint(-10, 10) for _ in range(rowsA * colsA)]

    # Expected res
    expected = transpose_and_scale_cpu(A_data, rowsA, colsA, scale_f)

    load_addrs = configMemory(A_data, rowsA, colsA, scale_f)

    runKernel(load_addrs, max_it=2_000_000)

    first_addr_out = first_addr + rowsA * colsA * 4
    result = getResult(first_addr_out, first_addr_out + rowsA * colsA * 4, rowsA*colsA)

    errors = sum(1 for i in range(len(expected))
                 if expected[i] != result[i])

# --------- OUTPUT MINIMAL POR TERMINAL ---------

with open(logfile, "r") as log:
    lines = log.readlines()
    for l in lines[-5:]:
        print(l.rstrip())

if errors > 0:
    print(f"Err: {errors}")
else:
    print("OK")
