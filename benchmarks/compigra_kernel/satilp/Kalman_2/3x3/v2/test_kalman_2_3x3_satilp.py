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
CGRA_N_ROWS = 3
CGRA_N_COLS = 3
SIZE = 60

# Address
first_addr = 128

# Ruta relativa desde la raíz del proyecto
kernel_name = f"benchmarks/compigra_kernel/satilp/Kalman_2/{CGRA_N_ROWS}x{CGRA_N_COLS}/v2/"
version = f"_{CGRA_N_COLS}_IJK{SIZE}"

# --------------------------------------------------------------------
# 3. FUNCIONES
# --------------------------------------------------------------------
def configMemory(A, Q, AP, NI):
    # Clear memory values
    kernel_clear_memory(kernel_name, version=version)
    
    # Config values            
    first_addr_A = first_addr
    first_addr_Q = first_addr_A + NI*NI*4
    first_addr_AP = first_addr_Q + NI*NI*4
    first_addr_APA = first_addr_AP + NI*NI*4
    first_addr_P = first_addr_APA + NI*NI*4

    config_vals = [[] for i in range(CGRA_N_COLS)]

    # void Kalman_2(int A[NI][NI], int Q[NI][NI],int AP[NI][NI], int APA[NI][NI], int P[NI][NI])
    # 0 : first_addr_A
    # 1 : first_addr_Q
    # 2 : first_addr_AP
    # 3 : first_addr_APA
    # 4 : first_addr_P

    config_vals[0] = [first_addr_AP, first_addr_Q] # 2, 1
    config_vals[1] = [] # 
    config_vals[2] = [first_addr_APA, first_addr_A, first_addr_P] # 3, 0, 4

    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i])*4
            
    # Load data
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_Q, Q, version=version)
    kernel_add_memory_region(kernel_name, first_addr_AP, AP, version=version)

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
Q = data["Q"]
AP = data["AP"]
NI = int(data["NI"])

# Expected results
APA_expected = data["APA_expected"]
P_expected = data["P_expected"]

print(f"Testing Karlman_2 sizes : {NI}")

load_addrs = configMemory(A, Q, AP, NI)

runKernel(load_addrs, max_it=20000000, printVal=0)
# estimatedConfigCycles(kernel_name, version)

# Get result from CGRA
first_addr_APA_res = first_addr + NI*NI*4*4
APA_result = getResult(first_addr_APA_res, NI*NI)

first_addr_P_res = first_addr + NI*NI*4*5
P_result = getResult(first_addr_P_res, NI*NI)

DEBUG = False

print("Check APA result:")
errors = 0
err_idx = []
for i in range(len(APA_expected)):
    if APA_expected[i] != APA_result[i]:
        errors += 1
        err_idx.append(i)
if errors > 0:
    print("Err: " + str(errors))
    if DEBUG:
        print("Expected: ")
        printAsMatrix(APA_expected, NI, NI)
        print("CGRA: ")
        printAsMatrix(APA_result, NI, NI)
        print("Errors are: Exp : CGRA")
        for i in err_idx:
            row = int(i/NI)
            col = i%NI
            print(f"Idx[{row}][{col}] {APA_expected[i]} : {APA_result[i]}")
else:
    print("OK")

print("Check P result:")
errors = 0
err_idx = []
for i in range(len(P_expected)):
    if P_expected[i] != P_result[i]:
        errors += 1
        err_idx.append(i)
if errors > 0:
    print("Err: " + str(errors))
    if DEBUG:
        print("Expected: ")
        printAsMatrix(P_expected, NI, NI)
        print("CGRA: ")
        printAsMatrix(P_result, NI, NI)
        print("Errors are: Exp : CGRA")
        for i in err_idx:
            row = int(i/NI)
            col = i%NI
            print(f"Idx[{row}][{col}] {P_expected[i]} : {P_result[i]}")
else:
    print("OK")