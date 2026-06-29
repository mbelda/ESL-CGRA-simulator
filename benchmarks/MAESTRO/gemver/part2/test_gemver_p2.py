#!/usr/bin/env python3
import os
import sys
import csv
import random
import logging
from pathlib import Path

# ------------------------------------------------------------------
#  DYNAMIC ROOT PATH INJECTION (4 levels up for MAESTRO layout)
# ------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

# repo/benchmarks/MAESTRO/gemver/part2/ -> repo/
repo_root = script_dir.parents[3] 
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from cgra import *
from kernels import *

# --------------------------------------------
#      GLOBAL CONFIG
# --------------------------------------------

DEBUG = 0
N = 16

# Global structural variables 
CGRA_N_ROWS = 4
CGRA_N_COLS = 4

# Address mapping
first_addr = 20000

# Employs local relative execution folder path
kernel_name = "./"
version = f"_meth"

# ------------------------------------------------------------------
#  LOGGING & TERMINAL REDIRECTION CONFIGURATION
# ------------------------------------------------------------------
log_filename = f"gemver_p2_{N}_output.log"

# Abrimos el archivo en modo escritura al iniciar el script
log_file = open(log_filename, "w")

class DualOutputMutedKernel:
    """Redirects standard print statements to both terminal and log file exactly as they are, 
    but intercepts and blocks deep internal kernel execution prints from the terminal."""
    def __init__(self, file_object):
        self.terminal = sys.__stdout__
        self.log_file = file_object
        self.mute_terminal = False

    def write(self, message):
        self.log_file.write(message)
        self.log_file.flush()
        
        if not self.mute_terminal:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Reemplazamos la salida estándar
sys.stdout = DualOutputMutedKernel(log_file)

# ------------------------------------
#           FUNCTIONS
# ------------------------------------
def configMemory(A, x, y, z, beta, N):
    kernel_clear_memory(kernel_name, version=version)
    
    # Mapeo de direcciones de memoria contiguas
    first_addr_A = first_addr
    first_addr_x = first_addr_A + (N * N * 4)
    first_addr_y = first_addr_x + (N * 4)
    first_addr_z = first_addr_y + (N * 4)
    
    # El resultado esperado es el vector x modificado
    first_addr_res = first_addr_x

    loopIit = int (N/16) -1
    loopJit = N

    # Configuración de constantes y punteros base para las columnas del CGRA
    # Recuerda que en hardware leerás la matriz A de forma transpuesta (por columnas)
    config_vals = [[] for i in range(CGRA_N_COLS)]

    # &A[0][0]      &z[1]       &x[2]       N
    # &x[4]         loopIit     &A[0][6]    &z[7]
    # N             &A[0][9]    &z[10]      &x[11]
    # &z[12]        &x[13]      N           &A[0][15]
    # --------------------------------------------
    # BETA          &y[0]       -           -
    # -             N           BETA        -
    # -             BETA        -           &y[0]
    # -             -           -           BETA

    config_vals[0] = [first_addr_A, first_addr_x + (4 * 4), N, first_addr_z + (12 * 4), beta]
    config_vals[1] = [first_addr_z + (1 * 4), loopIit, first_addr_A + (9 * 4), first_addr_x + (13 * 4), first_addr_y, N, beta]
    config_vals[2] = [first_addr_x + (2 * 4), first_addr_A + (6 * 4), first_addr_z + (10 * 4), N, beta]
    config_vals[3] = [N, first_addr_z + (7 * 4), first_addr_x + (11 * 4), first_addr_A + (15* 4), first_addr_y, beta]

    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i]) * 4
            
    # Carga de datos reales en la memoria del CGRA
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_x, x, version=version)
    kernel_add_memory_region(kernel_name, first_addr_y, y, version=version)
    kernel_add_memory_region(kernel_name, first_addr_z, z, version=version)

    return addr_config_loads, first_addr_res

def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])

def runKernel(load_addrs, max_it=1000, pr=["ROUT","INST"], printVal=1):
    sys.stdout.mute_terminal = True
    try:
        run(kernel_name, pr=pr, load_addrs=load_addrs, version=version, limit=max_it, printVal=printVal)
    finally:
        sys.stdout.mute_terminal = False

def getResult(first_addr, length):
    result = [0 for _ in range(length)]
    with open(os.path.join(kernel_name, f"memory_out{version}.csv"), 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            try:
                if (int(row[0]) >= first_addr) and (int(row[0]) < first_addr + length * 4):
                    result[int((int(row[0]) - first_addr) / 4)] = int(row[1])
            except ValueError:
                print("Error: Values in memory_out CSV file are not integers.")
    return result

# --------------------------------------------
#               DATA LOADING
# --------------------------------------------
data_path = os.path.join(kernel_name, "data", f"data_{N}.npz")
data = np.load(data_path)

A = data["A"]
x = data["x"]
y = data["y"]
z = data["z"]
beta = data["beta"]
x_expected = data["x_expected"]

print(f"Testing GEMVER Part 2 size:  {N}")
print(f"Full log execution details will write silently to: {log_filename}\n")

load_addrs, first_addr_res = configMemory(A, x, y, z, beta, N)

runKernel(load_addrs, max_it=2000000000, pr=["ROUT","R0", "INST"], printVal=1)

# El resultado esperado es el vector x (longitud N)
x_result = getResult(first_addr_res, N)

print("Check gemver_p2 X output result:")
errors = 0
err_idx = []
for i in range(len(x_expected)):
    if x_expected[i] != x_result[i]:
        errors += 1
        err_idx.append(i)

if errors > 0:
    print(f"Err: {errors} out of {N}.")
    
    print("Expected: ")
    printAsMatrix(x_expected, 1, N)
    print("CGRA: ")
    printAsMatrix(x_result, 1, N)
    if (DEBUG):
        print("Errors are: Exp : CGRA")
        for i in err_idx:
            print(f"Idx[{i}] {x_expected[i]} : {x_result[i]}")
else:
    print("OK")