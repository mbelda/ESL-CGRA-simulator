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

# repo/benchmarks/MAESTRO/atax/part1 -> repo/
repo_root = script_dir.parents[3] 
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from cgra import *
from kernels import *

# --------------------------------------------
#      GLOBAL CONFIG & PARAMETER PARSING
# --------------------------------------------

DEBUG = 1

# Lectura de M y N por parámetro de línea de comandos (Por defecto M=16, N=16)
if len(sys.argv) > 2:
    try:
        M = int(sys.argv[1])
        N = int(sys.argv[2])
    except ValueError:
        print(f"Error: M y N deben ser números enteros. Se recibió M='{sys.argv[1]}' y N='{sys.argv[2]}'.")
        sys.exit(1)
else:
    print("Aviso: No se proporcionaron parámetros completos (M N). Usando valores por defecto: M=16, N=16")
    M = 16
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
log_filename = f"atax_p1_{M}_{N}_output.log"

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
def configMemory(A, x, M, N):
    kernel_clear_memory(kernel_name, version=version)
    
    # Mapeo de direcciones contiguas para ATAX Parte 1 (Output es tmp)
    first_addr_A   = first_addr
    first_addr_x   = first_addr_A + (M * N * 4)
    first_addr_tmp = first_addr_x + (N * 4)
    
    # Puntero de retorno para la validación de la salida de memoria
    first_addr_res_tmp = first_addr_tmp

    # Parámetros e iteradores internos de los bucles
    loopIit = int(M/16) - 1
    loopJit = N-1

    # Estructura de almacenamiento por columna del CGRA
    config_vals = [[] for i in range(CGRA_N_COLS)]

    # Config vals
    # &A[0][0]       &A[1][0]       &A[2][0]       &A[3][0]
    # &A[4][0]       &A[5][0]       &A[6][0]       &A[7][0]
    # &A[8][0]       &A[9][0]       &A[10][0]      &A[11][0]
    # &A[12][0]      &A[13][0]      &A[14][0]      &A[15][0]
    # -----------------------------------------------------
    # N              &tmp[1]        &x[0]          
    # loopJit                       N              &tmp[7]
    # &x[0]          N              &tmp[10]         
    # &tmp[12]                      loopIit        N

    config_vals[0] = [first_addr_A, first_addr_A + (4 * N * 4), first_addr_A + (8 * N * 4), first_addr_A + (12 * N * 4), N, loopJit, first_addr_x, first_addr_tmp + (12 * 4)]
    config_vals[1] = [first_addr_A + (1 * N * 4), first_addr_A + (5 * N * 4), first_addr_A + (9 * N * 4), first_addr_A + (13 * N * 4), first_addr_tmp + (1 * 4), N]
    config_vals[2] = [first_addr_A + (2 * N * 4), first_addr_A + (6 * N * 4), first_addr_A + (10 * N * 4), first_addr_A + (14 * N * 4), first_addr_x, N, first_addr_tmp + (10 * 4), loopIit]
    config_vals[3] = [first_addr_A + (3 * N * 4), first_addr_A + (7 * N * 4), first_addr_A + (11 * N * 4), first_addr_A + (15 * N * 4), first_addr_tmp + (7 * 4), N]

    # Registro de las constantes empaquetadas en las regiones de carga
    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i]) * 4
            
    # Carga masiva de los arreglos en el simulador de memoria física
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_x, x, version=version)
    
    # Inicializamos el vector tmp a cero (Destino de la Parte 1)
    kernel_add_memory_region(kernel_name, first_addr_tmp, [0]*M, version=version)

    # --- IMPRESIÓN DE DIRECCIONES INICIALES COMPLETA ---
    print("------------------------------------------------")
    print("Direcciones de memoria iniciales unificadas (ATAX P1):")
    print(f"  Matriz A     (Inicio): {first_addr_A}")
    print(f"  Vector x     (Inicio): {first_addr_x}")
    print(f"  Vector tmp   (Inicio): {first_addr_tmp}")
    print("------------------------------------------------\n")

    return addr_config_loads, first_addr_res_tmp

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
data_path = os.path.join(kernel_name, "data", f"data_atax_p1_{M}_{N}.npz")
data = np.load(data_path)

A = data["A"]
x = data["x"]
tmp_expected = data["tmp_expected"]

print(f"Testing ATAX Part 1 size: M={M}, N={N}")
print(f"Full log execution details will write silently to: {log_filename}\n")

load_addrs, first_addr_res_tmp = configMemory(A, x, M, N)

# Lanzar simulación conjunta en el hardware integrado
runKernel(load_addrs, max_it=2000000000, pr=["ROUT", "R1", "R3", "INST"], printVal=1)

# --- CHECK VECTOR INTERMEDIO / SALIDA P1 (tmp: M) ---
tmp_result = getResult(first_addr_res_tmp, M)
print("Check ATAX tmp output result (Part 1):")
errors_tmp = sum(1 for i in range(len(tmp_expected)) if tmp_expected[i] != tmp_result[i])
if errors_tmp > 0:
    print(f"  Err: {errors_tmp} out of {M} in Vector tmp.")
    if DEBUG:
        print("Expected tmp:")
        printAsMatrix(tmp_expected, 1, M)
        print("CGRA tmp:")
        printAsMatrix(tmp_result, 1, M)
else:
    print("  OK")