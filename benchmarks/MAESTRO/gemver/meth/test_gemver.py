#!/usr/bin/env python3
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

# repo/benchmarks/MAESTRO/gemver-> repo/
repo_root = script_dir.parents[2] 
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from cgra import *
from kernels import *

# --------------------------------------------
#      GLOBAL CONFIG & PARAMETER PARSING
# --------------------------------------------

DEBUG = 1

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
log_filename = f"gemver_fused_all_{N}_output.log"

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
def configMemory(A, u1, v1, u2, v2, x, y, z, w, beta, alpha, N):
    kernel_clear_memory(kernel_name, version=version)
    
    # Mapeo de direcciones contiguas globales para todas las partes
    first_addr_A  = first_addr
    first_addr_u1 = first_addr_A + (N * N * 4)
    first_addr_v1 = first_addr_u1 + (N * 4)
    first_addr_u2 = first_addr_v1 + (N * 4)
    first_addr_v2 = first_addr_u2 + (N * 4)
    first_addr_x  = first_addr_v2 + (N * 4)
    first_addr_y  = first_addr_x + (N * 4)
    first_addr_z  = first_addr_y + (N * 4)
    first_addr_w  = first_addr_z + (N * 4)
    
    # Punteros de retorno para las validaciones individuales
    first_addr_res_A = first_addr_A
    first_addr_res_x = first_addr_x
    first_addr_res_w = first_addr_w

    # Parámetros e iteradores internos de los bucles
    loopIit_k1 = int(N/4) - 1
    loopJit_k1 = int(N/4) - 1
    loopIit_k2 = int(N/16) - 1
    loopIit_k3 = int(N/16) - 1

    # Estructura de almacenamiento por columna del CGRA
    config_vals = [[] for i in range(CGRA_N_COLS)]

    # === CONFIGURACIÓN LOCAL: KERNEL 1 ===
    k1_col0 = [first_addr_A, first_addr_u2 + (1 * 4), first_addr_A + (2 * N * 4), first_addr_u1 + (3 * 4), first_addr_v2, loopIit_k1, first_addr_v1]
    k1_col1 = [first_addr_u1, first_addr_A + (1 * N * 4) + (1 * 4), first_addr_u2 + (2 * 4), first_addr_A + (3 * N * 4) + (1 * 4), N, first_addr_v2 + (1 * 4), N, first_addr_v1 + (1 * 4)]
    k1_col2 = [first_addr_A + (2 * 4), first_addr_u1 + (1 * 4), first_addr_A + (2 * N * 4) + (2 * 4), first_addr_u2 + (3 * 4), first_addr_v1 + (2 * 4), loopJit_k1, first_addr_v2 + (2 * 4)]
    k1_col3 = [first_addr_u2, first_addr_A + (1 * N * 4) + (3 * 4), first_addr_u1 + (2 * 4), first_addr_A + (3 * N * 4) + (3 * 4), N, first_addr_v1 + (3 * 4), N, first_addr_v2 + (3 * 4)]

    # === CONFIGURACIÓN LOCAL: KERNEL 2 ===
    k2_col0 = [first_addr_A, first_addr_x + (4 * 4), N, first_addr_z + (12 * 4), beta]
    k2_col1 = [first_addr_z + (1 * 4), loopIit_k2, first_addr_A + (9 * 4), first_addr_x + (13 * 4), first_addr_y, N, beta]
    k2_col2 = [first_addr_x + (2 * 4), first_addr_A + (6 * 4), first_addr_z + (10 * 4), N, beta]
    k2_col3 = [N, first_addr_z + (7 * 4), first_addr_x + (11 * 4), first_addr_A + (15 * 4), first_addr_y, beta]

    # === CONFIGURACIÓN LOCAL: KERNEL 3 ===
    k3_col0 = [first_addr_A, first_addr_A + (4 * N * 4), first_addr_A + (8 * N * 4), first_addr_A + (12 * N * 4), alpha, loopIit_k3, first_addr_x, first_addr_w + (12 * 4)]
    k3_col1 = [first_addr_A + (1 * N * 4), first_addr_A + (5 * N * 4), first_addr_A + (9 * N * 4), first_addr_A + (13 * N * 4), N, first_addr_w + (5 * 4), alpha]
    k3_col2 = [first_addr_A + (2 * N * 4), first_addr_A + (6 * N * 4), first_addr_A + (10 * N * 4), first_addr_A + (14 * N * 4), first_addr_x, alpha, first_addr_w + (10 * 4)]
    k3_col3 = [first_addr_A + (3 * N * 4), first_addr_A + (7 * N * 4), first_addr_A + (11 * N * 4), first_addr_A + (15 * N * 4), first_addr_w + (3 * 4), N, alpha]

    # Fusión secuencial ordenada (K1 -> K2 -> K3)
    config_vals[0] = k1_col0 + k2_col0 + k3_col0
    config_vals[1] = k1_col1 + k2_col1 + k3_col1
    config_vals[2] = k1_col2 + k2_col2 + k3_col2
    config_vals[3] = k1_col3 + k2_col3 + k3_col3

    # Registro de las constantes empaquetadas en las regiones de carga
    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i]) * 4
            
    # Carga masiva de los arreglos en el simulador de memoria física
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_u1, u1, version=version)
    kernel_add_memory_region(kernel_name, first_addr_v1, v1, version=version)
    kernel_add_memory_region(kernel_name, first_addr_u2, u2, version=version)
    kernel_add_memory_region(kernel_name, first_addr_v2, v2, version=version)
    kernel_add_memory_region(kernel_name, first_addr_x, x, version=version)
    kernel_add_memory_region(kernel_name, first_addr_y, y, version=version)
    kernel_add_memory_region(kernel_name, first_addr_z, z, version=version)
    kernel_add_memory_region(kernel_name, first_addr_w, w, version=version)

    # --- IMPRESIÓN DE DIRECCIONES INICIALES COMPLETA ---
    print("------------------------------------------------")
    print("Direcciones de memoria iniciales unificadas (P1 + P2 + P3):")
    print(f"  Matriz A   (Inicio): {first_addr_A}")
    print(f"  Vector u1  (Inicio): {first_addr_u1}")
    print(f"  Vector v1  (Inicio): {first_addr_v1}")
    print(f"  Vector u2  (Inicio): {first_addr_u2}")
    print(f"  Vector v2  (Inicio): {first_addr_v2}")
    print(f"  Vector x   (Inicio): {first_addr_x}")
    print(f"  Vector y   (Inicio): {first_addr_y}")
    print(f"  Vector z   (Inicio): {first_addr_z}")
    print(f"  Vector w   (Inicio): {first_addr_w}")
    print("------------------------------------------------\n")

    return addr_config_loads, first_addr_res_A, first_addr_res_x, first_addr_res_w

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
x = data["x"]
y = data["y"]
z = data["z"]
w = data["w"]
beta = data["beta"]
alpha = data["alpha"]

A_expected = data["A_expected"]
x_expected = data["x_expected"]
w_expected = data["w_expected"]

print(f"Testing FULL FUSED GEMVER (Part 1, 2 & 3) size: {N}")
print(f"Full log execution details will write silently to: {log_filename}\n")

load_addrs, first_addr_res_A, first_addr_res_x, first_addr_res_w = configMemory(
    A, u1, v1, u2, v2, x, y, z, w, beta, alpha, N
)

# Lanzar simulación conjunta en el hardware integrado
runKernel(load_addrs, max_it=2000000000, pr=["ROUT", "R0", "R1", "R2", "R3", "INST"], printVal=1)

# --- CHECK PARTE 1 (Matriz A: N*N) ---
A_result = getResult(first_addr_res_A, N * N)
print("Check gemver_p1 A output result:")
errors_A = sum(1 for i in range(len(A_expected)) if A_expected[i] != A_result[i])
if errors_A > 0:
    print(f"  Err: {errors_A} out of {N * N} in Matrix A.")
else:
    print("  OK")

# --- CHECK PARTE 2 (Vector x: N) ---
x_result = getResult(first_addr_res_x, N)
print("Check gemver_p2 X output result:")
errors_x = sum(1 for i in range(len(x_expected)) if x_expected[i] != x_result[i])
if errors_x > 0:
    print(f"  Err: {errors_x} out of {N} in Vector x.")
else:
    print("  OK")

# --- CHECK PARTE 3 (Vector w: N) ---
w_result = getResult(first_addr_res_w, N)
print("Check gemver_p3 W output result:")
errors_w = sum(1 for i in range(len(w_expected)) if w_expected[i] != w_result[i])
if errors_w > 0:
    print(f"  Err: {errors_w} out of {N} in Vector w.")
    if DEBUG:
        print("Expected W:")
        printAsMatrix(w_expected, 1, N)
        print("CGRA W:")
        printAsMatrix(w_result, 1, N)
else:
    print("  OK")