import os
import sys
import csv
import random
import logging
from pathlib import Path

# ------------------------------------------------------------------
#  DYNAMIC ROOT PATH INJECTION (3 levels up for MAESTRO layout)
# ------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

# repo/benchmarks/MAESTRO/gesummv/ -> repo/
repo_root = script_dir.parents[2] 
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from cgra import *
from kernels import *

# --------------------------------------------
#      GLOBAL CONFIG
# --------------------------------------------

DEBUG = 0
N = 24

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
log_filename = f"gesummv_{N}_output.log"

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
        # Escribe de forma exacta y literal en el archivo log (mantiene tabulaciones, saltos, etc.)
        self.log_file.write(message)
        self.log_file.flush()
        
        # Muestra en la terminal física solo si no estamos ejecutando runKernel
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
def configMemory(A, B, x, y, alpha, beta, N):
    kernel_clear_memory(kernel_name, version=version)
    
    first_addr_A = first_addr
    first_addr_B = first_addr_A + (N * N * 4)
    first_addr_x = first_addr_B + (N * N * 4)
    first_addr_y = first_addr_x + (N * 4)
    
    first_addr_res = first_addr_y

    loopIit = int(N/8)
    loopJit = N

    # Config vals
    # &A[0][0]     alpha        &y[2]      loopJit
    # -            &B[0][0]     &x[0]      beta
    # &y[4]        beta         -          &B[0][0]
    # &x[0]        loopIit      &A[0][0]   alpha
    # ---------------------------------------------
    # 4*N          -            -          -
    # -            4*N          -          -
    # -            -            -          4*N
    # -            -            4*N        -



    config_vals = [[] for i in range(CGRA_N_COLS)]

    config_vals[0] = [first_addr_A, first_addr_y + 16, first_addr_x, 4*N] 
    config_vals[1] = [alpha, first_addr_B, beta, loopIit, 4*N] 
    config_vals[2] = [first_addr_y + 8, first_addr_x, first_addr_A, 4*N] 
    config_vals[3] = [loopJit, beta, first_addr_B, alpha, 4*N]

    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i]) * 4
            
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_B, B, version=version)
    kernel_add_memory_region(kernel_name, first_addr_x, x, version=version)
    kernel_add_memory_region(kernel_name, first_addr_y, y, version=version)

    return addr_config_loads, first_addr_res

def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])

def runKernel(load_addrs, max_it=1000, pr=["ROUT","INST"], printVal=1):
    # Tell our redirector to suppress terminal output for the underlying C/Python runner
    sys.stdout.mute_terminal = True
    try:
        run(kernel_name, pr=pr, load_addrs=load_addrs, version=version, limit=max_it, printVal=printVal)
    finally:
        # Unmute terminal as soon as the core simulation block finishes
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
B = data["B"]
x = data["x"]
y = data["y"]
alpha = data["alpha"]
beta = data["beta"]
y_expected = data["y_expected"]

print(f"Testing GESUMMV size:  {N}")
print(f"Full log execution details will write silently to: {log_filename}\n")

load_addrs, first_addr_res = configMemory(A, B, x, y, alpha, beta, N)

runKernel(load_addrs, max_it=2000000000, pr=["ROUT","R0", "INST"], printVal=1)

y_result = getResult(first_addr_res, N)

print("Check gesummv Y output result:")
errors = 0
err_idx = []
for i in range(len(y_expected)):
    if y_expected[i] != y_result[i]:
        errors += 1
        err_idx.append(i)

if errors > 0:
    print(f"Err: {errors} out of {N}.")
    
    print("Expected: ")
    printAsMatrix(y_expected, 1, N)
    print("CGRA: ")
    printAsMatrix(y_result, 1, N)
    if (DEBUG):
        print("Errors are: Exp : CGRA")
        for i in err_idx:
            print(f"Idx[{i}] {y_expected[i]} : {y_result[i]}")
else:
    print("OK")