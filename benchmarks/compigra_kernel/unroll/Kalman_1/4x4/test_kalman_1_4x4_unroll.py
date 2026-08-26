import os
import sys
from pathlib import Path
import random
import csv
import numpy as np

# --------------------------------------------------------------------
# 1. RESOLUCIÓN DE RUTAS Y CONFIGURACIÓN DEL CWD
# --------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Busca la raíz del proyecto (donde existe cgra.py)
ROOT_DIR = next((p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "cgra.py").exists()), None)

if ROOT_DIR:
    # 1. Añade la raíz del proyecto al path para importar cgra y kernels
    sys.path.insert(0, str(ROOT_DIR))
    # 2. Cambia el directorio de trabajo a la raíz del proyecto
    os.chdir(ROOT_DIR)
else:
    raise RuntimeError("No se encontró el archivo 'cgra.py' en ningún directorio superior.")

# Importación de los módulos del simulador
from cgra import *
from kernels import *

# --------------------------------------------------------------------
# 2. CONFIGURACIÓN DEL BENCHMARK
# --------------------------------------------------------------------
CGRA_N_ROWS = 4
CGRA_N_COLS = 4
SIZE = 64

# Address
first_addr = 20000

# Benchmark (Ruta relativa desde la raíz)
kernel_name = f"benchmarks/compigra_kernel/unroll/Kalman_1/{CGRA_N_ROWS}x{CGRA_N_COLS}/"
version = f"_out_{CGRA_N_COLS}_I{SIZE}_J{SIZE}_K{SIZE}"

# --------------------------------------------------------------------
# 3. FUNCIONES
# --------------------------------------------------------------------
def configMemory(A, x, P, NI):
    # Clear memory values
    kernel_clear_memory(kernel_name, version=version)
    
    # Config values            
    first_addr_A = first_addr
    first_addr_x = first_addr_A + NI*NI*4
    first_addr_P = first_addr_x + NI*4
    first_addr_Ax = first_addr_P + NI*NI*4
    first_addr_AP = first_addr_Ax + NI*4

    config_vals = [[] for i in range(CGRA_N_COLS)]

    # void Kalman_1(int A[NI][NI], int x[NI], int P[NI][NI], int Ax[NI], int AP[NI][NI])
    # 0 : first_addr_A
    # 1 : first_addr_x
    # 2 : first_addr_P
    # 3 : first_addr_Ax
    # 4 : first_addr_AP

    config_vals[0] = [first_addr_x, first_addr_A] # 1, 0
    config_vals[1] = [first_addr_x, first_addr_Ax, first_addr_AP, first_addr_P] # 1, 3, 4, 2 
    config_vals[2] = [first_addr_A] # 0
    config_vals[3] = [] # 

    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i])*4
            
    # Load data
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_x, x, version=version)
    kernel_add_memory_region(kernel_name, first_addr_P, P, version=version)

    # Config data address for direct loads
    return addr_config_loads

def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])

def runKernel(load_addrs, max_it=1000, pr=["ROUT","INST"], printVal=1):
    # Run kernel
    run(kernel_name, pr=pr, load_addrs=load_addrs, version=version, limit=max_it, printVal=printVal)

def getResult(first_addr, length):
    result = [0 for _ in range(length)]
    with open(kernel_name + "/memory_out" + version + ".csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            try:
                if (int(row[0]) >= first_addr) and (int(row[0]) < first_addr + length*4):
                    result[int((int(row[0]) - first_addr)/4)] = int(row[1])
            except ValueError:
                print("Error: Values in memory_out CSV file are not integers.")
    return result

# --------------------------------------------------------------------
# 4. DATA & EJECUCIÓN DEL TEST
# --------------------------------------------------------------------
data = np.load(kernel_name + f"data/data_{SIZE}.npz")

A = data["A"]
x = data["x"]
P = data["P"]
NI = int(data["NI"])

# Expected results
Ax_expected = data["Ax_expected"]
AP_expected = data["AP_expected"]

print(f"Testing PCA sizes : {NI}")

load_addrs = configMemory(A, x, P, NI)

runKernel(load_addrs, max_it=20000000000, printVal=0)
# estimatedConfigCycles(kernel_name, version)

# Get result from CGRA
first_addr_Ax_res = first_addr + NI*NI*4*2 + NI*4
Ax_result = getResult(first_addr_Ax_res, NI)

first_addr_AP_res = first_addr + NI*NI*4*2 + 2*NI*4
AP_result = getResult(first_addr_AP_res, NI*NI)

# Check result correctness
DEBUG = 0

print("Check Ax result:")
errors = 0
err_idx = []
for i in range(len(Ax_expected)):
    if Ax_expected[i] != Ax_result[i]:
        errors += 1
        err_idx.append(i)
if errors > 0:
    print("Err: " + str(errors))
    if DEBUG:
        print("Expected: ")
        printAsMatrix(Ax_expected, 1, NI)
        print("CGRA: ")
        printAsMatrix(Ax_result, 1, NI)
        print("Errors are: Exp : CGRA")
        for i in err_idx:
            print(f"Idx[{i}] {Ax_expected[i]} : {Ax_result[i]}")
else:
    print("OK")

print("Check AP result:")
errors = 0
err_idx = []
for i in range(len(AP_expected)):
    if AP_expected[i] != AP_result[i]:
        errors += 1
        err_idx.append(i)
if errors > 0:
    print("Err: " + str(errors))
    if DEBUG:
        print("Expected: ")
        printAsMatrix(AP_expected, NI, NI)
        print("CGRA: ")
        printAsMatrix(AP_result, NI, NI)
        print("Errors are: Exp : CGRA")
        for i in err_idx:
            row = int(i/NI)
            col = i%NI
            print(f"Idx[{row}][{col}] {AP_expected[i]} : {AP_result[i]}")
else:
    print("OK")