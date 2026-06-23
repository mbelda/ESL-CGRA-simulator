import sys
import argparse
import csv
from contextlib import redirect_stdout
import random

from cgra import *
from kernels import *

kernel_name = "benchmarks/disco-vs-oe/add"
version = ""

# Global variables
CGRA_N_ROWS = 4
CGRA_N_COLS = 4
first_addr = 20000
DEBUG = False

def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])


def configMemory(A_data, B_data, rowsA, colsA):
    kernel_clear_memory(kernel_name, version=version)

    nItLoop = rowsA*colsA

    first_addr_A = first_addr
    first_addr_B = first_addr_A + rowsA * colsA * 4

    config_vals_col0 = [first_addr_A, first_addr_A + 4*4, first_addr_A + 2*4*4, first_addr_A + 3*4*4,
                        nItLoop,      first_addr_B + 4*4, first_addr_B + 2*4*4, first_addr_B + 3*4*4]

    config_vals_col1 = [first_addr_A + 4, first_addr_A + 4*4 +4, first_addr_A + 2*4*4 +4, first_addr_A + 3*4*4 +4,
                        first_addr_B + 4, first_addr_B + 4*4 +4, first_addr_B + 2*4*4 +4, first_addr_B + 3*4*4 +4]

    config_vals_col2 = [first_addr_A + 8, first_addr_A + 4*4 +8, first_addr_A + 2*4*4 +8, first_addr_A + 3*4*4 +8,
                        first_addr_B + 8, first_addr_B + 4*4 +8, first_addr_B + 2*4*4 +8, first_addr_B + 3*4*4 +8]

    config_vals_col3 = [first_addr_A + 12, first_addr_A + 4*4 +12, first_addr_A + 2*4*4 +12, first_addr_A + 3*4*4 +12,
                        first_addr_B + 12, first_addr_B + 4*4 +12, first_addr_B + 2*4*4 +12, first_addr_B + 3*4*4 +12]

    addr = 0
    for cfg in [config_vals_col0, config_vals_col1,
                config_vals_col2, config_vals_col3]:
        kernel_add_memory_region(kernel_name, addr, cfg, version=version)
        addr += len(cfg) * 4

    kernel_add_memory_region(kernel_name, first_addr_A, A_data, version=version)
    kernel_add_memory_region(kernel_name, first_addr_B, B_data, version=version)

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


def add_cpu(A_data, B_data, rowsA, colsA):
    out = [0 for _ in range(rowsA * colsA)]
    for i in range(rowsA*colsA):
        out[i] = A_data[i] + B_data[i]
    return out


# ---------------------- MAIN ----------------------

parser = argparse.ArgumentParser()
parser.add_argument("rowsA", type=int)
parser.add_argument("colsA", type=int)
args = parser.parse_args()

rowsA, colsA = args.rowsA, args.colsA
logfile = f"out_add_{rowsA}x{colsA}.log"

with open(logfile, "w") as log, redirect_stdout(log):

    A_data = [random.randint(-10, 10) for _ in range(rowsA * colsA)]
    B_data = [random.randint(-10, 10) for _ in range(rowsA * colsA)]

    # Expected res
    expected = add_cpu(A_data, B_data, rowsA, colsA)

    load_addrs = configMemory(A_data, B_data, rowsA, colsA)

    runKernel(load_addrs, max_it=2_000_000)

    first_addr_C = first_addr
    result = getResult(first_addr, first_addr + rowsA * colsA * 4, rowsA*colsA)

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
