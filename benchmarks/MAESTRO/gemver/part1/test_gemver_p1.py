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

# repo/benchmarks/MAESTRO/gemver/part1/ -> repo/
repo_root = script_dir.parents[3] 
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from cgra import *
from kernels import *

# --------------------------------------------
#      GLOBAL CONFIG & PARAMETER PARSING
# --------------------------------------------

DEBUG = 0

# Lectura de N por parámetro de línea de comandos (Por defecto 16 si no se especifica)
if len(sys.argv) > 1:
    try:
        N = int(sys.argv[1])
    except ValueError:
        print(f"Error: El parámetro N debe ser un número entero. Se recibió '{sys.argv[1]}'.")
        sys.exit(1)
else:
    print("Aviso: No se proporcionó el parámetro N. Usando valor por defecto: 16")
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
log_filename = f"gemver_p1_{N}_output.log"

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
def configMemory(A, u1, v1, u2, v2, N):
    kernel_clear_memory(kernel_name, version=version)
    
    # Mapeo de direcciones de memoria contiguas
    first_addr_A  = first_addr
    first_addr_u1 = first_addr_A + (N * N * 4)
    first_addr_v1 = first_addr_u1 + (N * 4)
    first_addr_u2 = first_addr_v1 + (N * 4)
    first_addr_v2 = first_addr_u2 + (N * 4)
    
    # En la parte 1 el resultado esperado es la propia matriz A modificada
    first_addr_res = first_addr_A

    loopIit = int(N/4) -1
    loopJit = int(N/4) -1

    # Configuración de las regiones/valores de constantes para las columnas del CGRA
    config_vals = [[] for i in range(CGRA_N_COLS)]

    # &A[0][0]      &u1[0]       &A[0][2]       &u2[0]
    # &u2[1]        &A[1][1]     &u1[1]         &A[1][3]
    # &A[2][0]      &u2[2]       &A[2][2]       &u1[2]
    # &u1[3]        &A[3][1]     &u2[3]         &A[3][3]
    # --------------------------------------------
    # &v2[0]        N            &v1[2]         N
    # loopIit       &v2[1]       loopJit        &v1[3]
    # &v1[0]        N            &v2[2]         N
    # -             &v1[1]       -              &v2[3]

    config_vals[0] = [first_addr_A, first_addr_u2 + (1 * 4), first_addr_A + (2 * N * 4), first_addr_u1 + (3 * 4), first_addr_v2, loopIit, first_addr_v1]
    config_vals[1] = [first_addr_u1, first_addr_A + (1 * N * 4) + (1 * 4), first_addr_u2 + (2 * 4), first_addr_A + (3 * N * 4) + (1 * 4), N, first_addr_v2 + (1 * 4), N, first_addr_v1 + (1 * 4)]
    config_vals[2] = [first_addr_A + (2 * 4), first_addr_u1 + (1 * 4), first_addr_A + (2 * N * 4) + (2 * 4), first_addr_u2 + (3 * 4), first_addr_v1 + (2 * 4), loopJit, first_addr_v2 + (2 * 4)]
    config_vals[3] = [first_addr_u2, first_addr_A + (1 * N * 4) + (3 * 4), first_addr_u1 + (2 * 4), first_addr_A + (3 * N * 4) + (3 * 4), N, first_addr_v1 + (3 * 4), N, first_addr_v2 + (3 * 4)]

    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i]) * 4
            
    # Carga de datos reales en el simulador de memoria del CGRA
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_u1, u1, version=version)
    kernel_add_memory_region(kernel_name, first_addr_v1, v1, version=version)
    kernel_add_memory_region(kernel_name, first_addr_u2, u2, version=version)
    kernel_add_memory_region(kernel_name, first_addr_v2, v2, version=version)

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
u1 = data["u1"]
v1 = data["v1"]
u2 = data["u2"]
v2 = data["v2"]
A_expected = data["A_expected"]

print(f"Testing GEMVER Part 1 size:  {N}")
print(f"Full log execution details will write silently to: {log_filename}\n")

load_addrs, first_addr_res = configMemory(A, u1, v1, u2, v2, N)

runKernel(load_addrs, max_it=2000000000, pr=["ROUT", "R1", "INST"], printVal=1)

# El resultado esperado es una matriz NxN (N*N elementos)
A_result = getResult(first_addr_res, N * N)

print("Check gemver_p1 A output result:")
errors = 0
err_idx = []
for i in range(len(A_expected)):
    if A_expected[i] != A_result[i]:
        errors += 1
        err_idx.append(i)

if errors > 0:
    print(f"Err: {errors} out of {N * N}.")
    
    print("Expected (First 2 rows):")
    printAsMatrix(A_expected, N, N)
    print("CGRA (First 2 rows):")
    printAsMatrix(A_result, N, N)

    if (DEBUG):
        print("\nErrors are: Exp : CGRA")
        for i in err_idx[:50]:
            print(f"Idx[{i}] {A_expected[i]} : {A_result[i]}")
else:
    print("OK")