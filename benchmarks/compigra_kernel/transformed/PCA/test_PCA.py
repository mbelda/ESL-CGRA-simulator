#!/usr/bin/env python3
import sys
import csv
import random
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuración de rutas
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.name != "benchmarks" and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

kernel_name = str(SCRIPT_DIR.relative_to(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Importación de módulos del proyecto
# -----------------------------------------------------------------------------
from cgra import *
from kernels import *

# -----------------------------------------------------------------------------
# Variables globales y de configuración
# -----------------------------------------------------------------------------
dim = 24
CGRA_N_ROWS = 4
CGRA_N_COLS = 4
version = f"_{CGRA_N_ROWS}x{CGRA_N_COLS}_{dim}"

first_addr = 2000

# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------
def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])

def configMemory(inputX_data, mu_data, Xc_data, XcT_data, C_data, NI, NJ):
    kernel_clear_memory(".", version=version)
    
    # Cálculo de direcciones iniciales (4 bytes por entero)
    first_addr_inputX = first_addr
    first_addr_mu     = first_addr_inputX + (NI * NJ * 4)
    first_addr_Xc     = first_addr_mu     + (NJ * 4)
    first_addr_XcT    = first_addr_Xc     + (NI * NJ * 4)
    first_addr_C      = first_addr_XcT    + (NJ * NI * 4)

    # Configuración de columnas provista
    config_cols = [
        [],
        [first_addr_Xc, first_addr_C],
        [first_addr_mu, first_addr_Xc],
        [first_addr_inputX]
    ]
    
    load_addrs = []
    current_addr = 0
    
    for config_vals in config_cols:
        load_addrs.append(current_addr)
        kernel_add_memory_region(".", current_addr, config_vals, version=version)
        current_addr += len(config_vals) * 4
    
    # Carga de datos de entrada y salida
    data_regions = [
        (first_addr_inputX, inputX_data),
        (first_addr_mu, mu_data),
        (first_addr_Xc, Xc_data),
        (first_addr_XcT, XcT_data),
        (first_addr_C, C_data)
    ]
    
    for addr, data in data_regions:
        kernel_add_memory_region(".", addr, data, version=version)
    
    return load_addrs

def runKernel(load_addrs, max_it=1000, printVal=0):
    run(".", pr=["ROUT", "R0", "R1", "INST"], load_addrs=load_addrs, version=version, limit=max_it, printVal=printVal)

def getResult(start_addr, end_addr, rows, cols):
    result = [0 for _ in range(rows * cols)]
    csv_file_path = Path(f"memory_out{version}.csv")
    
    with open(csv_file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            try:
                addr = int(row[0])
                if start_addr <= addr < end_addr:
                    result[int((addr - start_addr) / 4)] = int(row[1])
            except ValueError:
                print("Error: Values in memory_out CSV file are not integers.")
    return result

def pca_cpu(inputX_data, mu_data, NI, NJ):
    Xc = [0 for _ in range(NI * NJ)]
    C  = [0 for _ in range(NJ * NJ)]
    
    # Center
    for i in range(NI):
        for j in range(NJ):
            Xc[i * NJ + j] = inputX_data[i * NJ + j] - mu_data[j]
            
    # Matmul: C = Xc^T * Xc
    for i in range(NJ):
        for j in range(NJ):
            sum_val = 0
            for k in range(NI):
                sum_val += Xc[k * NJ + i] * Xc[k * NJ + j]
            C[i * NJ + j] = sum_val
            
    return C

# -----------------------------------------------------------------------------
# Flujo Principal
# -----------------------------------------------------------------------------
def main():
    print(f"[-] Ejecutando kernel: {kernel_name}")

    NI = dim
    NJ = dim

    # Generar datos aleatorios de entrada
    inputX_data = [random.randint(-10, 10) for _ in range(NI * NJ)]
    mu_data     = [random.randint(-10, 10) for _ in range(NJ)]
    
    # Inicializar búferes de trabajo/salida a cero
    Xc_data  = [0 for _ in range(NI * NJ)]
    XcT_data = [0 for _ in range(NJ * NI)]
    C_data   = [0 for _ in range(NJ * NJ)]

    # Copias de seguridad para la CPU
    inputX_cpy = inputX_data.copy()
    mu_cpy     = mu_data.copy()

    # Configurar memoria y ejecutar kernel
    load_addrs = configMemory(inputX_data, mu_data, Xc_data, XcT_data, C_data, NI, NJ)
    runKernel(load_addrs, max_it=200000, printVal=0)

    # Cálculo de la dirección base y límite de 'C' (resultado final)
    first_addr_inputX = first_addr
    first_addr_mu     = first_addr_inputX + (NI * NJ * 4)
    first_addr_Xc     = first_addr_mu     + (NJ * 4)
    first_addr_XcT    = first_addr_Xc     + (NI * NJ * 4)
    first_addr_C      = first_addr_XcT    + (NJ * NI * 4)
    end_addr_C        = first_addr_C      + (NJ * NJ * 4)

    # Obtener resultado del CGRA desde la región de 'C'
    result = getResult(first_addr_C, end_addr_C, NJ, NJ)

    # Obtener resultado de referencia en CPU
    expected_res = pca_cpu(inputX_cpy, mu_cpy, NI, NJ)

    # Comprobar diferencias
    errors = 0
    for i in range(len(expected_res)):
        if expected_res[i] != result[i]:
            errors += 1

    if errors > 0:
        print(f"Err: {errors}")
        print("Expected: ")
        printAsMatrix(expected_res, NJ, NJ)
        print("CGRA: ")
        printAsMatrix(result, NJ, NJ)
    else:
        print("OK")

if __name__ == "__main__":
    main()