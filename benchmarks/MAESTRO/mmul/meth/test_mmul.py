import os
import sys
import csv
import random
import logging
import argparse
from pathlib import Path

# ------------------------------------------------------------------
#  DYNAMIC ROOT PATH INJECTION (3 levels up for MAESTRO layout)
# ------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

# repo/benchmarks/MAESTRO/mmul/ -> repo/
repo_root = script_dir.parents[2] 
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from cgra import *
from kernels import *

# --------------------------------------------
#      GLOBAL CONFIG & ARGUMENT PARSING
# --------------------------------------------
DEBUG = 0

parser = argparse.ArgumentParser(description="Script para ejecutar MMUL con dimensiones dinámicas.")
parser.add_argument('--NI', type=int, help='Dimensión NI')
parser.add_argument('--NJ', type=int, help='Dimensión NJ')
parser.add_argument('--NK', type=int, help='Dimensión NK')
parser.add_argument('-N', '--N', type=int, help='Dimensión única para NI, NJ y NK')

args = parser.parse_args()

if args.N is not None:
    NI = args.N
    NJ = args.N
    NK = args.N
else:
    NI = args.NI if args.NI is not None else 16
    NJ = args.NJ if args.NJ is not None else 16
    NK = args.NK if args.NK is not None else 16

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
log_filename = f"mmul_{NI}x{NJ}_output.log"

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
def configMemory(A, B, NI, NK, NJ):
    kernel_clear_memory(kernel_name, version=version)
    
    # Tamaños en bytes (asumiendo enteros de 4 bytes)
    size_A = NI * NK * 4
    size_B = NK * NJ * 4

    first_addr_A = first_addr
    first_addr_B = first_addr_A + size_A
    
    # El resultado se escribe directamente en la dirección consecutiva a B
    first_addr_res = first_addr_B + size_B  

    # Direcciones de memoria calculadas para el mapeo
    addr_A_0_0 = first_addr_A
    addr_A_1_0 = first_addr_A + (1 * NK * 4)
    addr_A_2_0 = first_addr_A + (2 * NK * 4)
    addr_A_3_0 = first_addr_A + (3 * NK * 4)

    addr_B_0_0 = first_addr_B
    addr_B_0_1 = first_addr_B + 4
    addr_B_0_2 = first_addr_B + 8
    addr_B_0_3 = first_addr_B + 12

    addr_C_0_2 = first_addr_res + 8
    addr_C_1_3 = first_addr_res + (1 * NJ * 4) + 12
    addr_C_2_0 = first_addr_res + (2 * NJ * 4)
    addr_C_3_1 = first_addr_res + (3 * NJ * 4) + 4

    # Iteraciones de los bucles (decrementadas en 1)
    loopIit = int(NI / 4) - 1
    loopJit = int(NJ / 4) - 1
    loopKit = NK - 1

    # Config vals estructurado en 4 columnas (sin alpha ni beta)
    config_vals = [[] for i in range(CGRA_N_COLS)]

    config_vals[0] = [addr_A_0_0, loopIit, addr_C_2_0, addr_B_0_0, NK]
    config_vals[1] = [addr_B_0_1, addr_A_1_0, addr_C_3_1, NK, NJ]
    config_vals[2] = [addr_C_0_2, addr_B_0_2, addr_A_2_0, loopKit, NJ, NK]
    config_vals[3] = [loopJit, addr_C_1_3, addr_B_0_3, addr_A_3_0, NJ]

    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i]) * 4
            
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_B, B, version=version)

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
data_path = os.path.join(kernel_name, "data", f"data_{NI}.npz")
data = np.load(data_path)

A = data["A"]
B = data["B"]
C_expected = data["C_expected"]  

total_elements_C = NI * NJ

# --- CÁLCULO PREVIO DE DIRECCIONES PARA MOSTRAR AL INICIO ---
size_A = NI * NK * 4
size_B = NK * NJ * 4

first_addr_A = first_addr
first_addr_B = first_addr_A + size_A
first_addr_res = first_addr_B + size_B

print("====================================================")
print("             CGRA MEMORY MAPPING DETAILS            ")
print("====================================================")
print(f"Base Address (first_addr) : {first_addr}")
print(f"Matrix A Base Address     : {first_addr_A}")
print(f"Matrix B Base Address     : {first_addr_B}")
print(f"Result C Base Address     : {first_addr_res}")
print("====================================================\n")

print(f"Testing MMUL size: {NI}x{NK}x{NJ}")
print(f"Full log execution details will write silently to: {log_filename}\n")

load_addrs, first_addr_res = configMemory(A, B, NI, NK, NJ)

runKernel(load_addrs, max_it=2000000000, pr=["ROUT", "INST"], printVal=1)

C_result = getResult(first_addr_res, total_elements_C)

print("Check MMUL C output result:")
errors = 0
err_idx = []
for i in range(len(C_expected)):
    if C_expected[i] != C_result[i]:
        errors += 1
        err_idx.append(i)

if errors > 0:
    print(f"Err: {errors} out of {total_elements_C}.")
    if (DEBUG):
        print("Expected: ")
        printAsMatrix(C_expected, NI, NJ)
        print("CGRA: ")
        printAsMatrix(C_result, NI, NJ)
else:
    print("OK")