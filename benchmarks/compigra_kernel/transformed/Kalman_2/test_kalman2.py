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

def configMemory(A_data, Q_data, AP_data, APA_data, P_data, NI):
    kernel_clear_memory(".", version=version)
    
    # Cálculo de direcciones iniciales (4 bytes por entero)
    first_addr_A   = first_addr
    first_addr_Q   = first_addr_A   + (NI * NI * 4)
    first_addr_AP  = first_addr_Q   + (NI * NI * 4)
    first_addr_APA = first_addr_AP  + (NI * NI * 4)
    first_addr_P   = first_addr_APA + (NI * NI * 4)

    # Configuración de columnas provista
    config_cols = [
        [],
        [first_addr_Q, first_addr_APA],
        [first_addr_A],
        [first_addr_AP, first_addr_P]
    ]
    
    load_addrs = []
    current_addr = 0
    
    for config_vals in config_cols:
        load_addrs.append(current_addr)
        kernel_add_memory_region(".", current_addr, config_vals, version=version)
        current_addr += len(config_vals) * 4
    
    # Carga de datos de entrada/salida
    data_regions = [
        (first_addr_A, A_data),
        (first_addr_Q, Q_data),
        (first_addr_AP, AP_data),
        (first_addr_APA, APA_data),
        (first_addr_P, P_data)
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

def kalman_2_cpu(A_data, Q_data, AP_data, APA_data, P_data, NI):
    expected_APA = [0 for _ in range(NI * NI)]
    expected_P   = [0 for _ in range(NI * NI)]
    
    # APA = AP * AT
    for i in range(NI):
        for j in range(NI):
            sum_val = 0
            # Fíjate en el acceso a A_data: A[j][k] (simula matriz transpuesta)
            for k in range(NI):
                sum_val += AP_data[i * NI + k] * A_data[j * NI + k]
            expected_APA[i * NI + j] = sum_val

    # P = APA + Q
    for i in range(NI):
        for j in range(NI):
            expected_P[i * NI + j] = expected_APA[i * NI + j] + Q_data[i * NI + j]
            
    return expected_P

# -----------------------------------------------------------------------------
# Flujo Principal
# -----------------------------------------------------------------------------
def main():
    print(f"[-] Ejecutando kernel: {kernel_name}")

    NI = dim

    # Generar datos aleatorios para matrices NxN
    A_data   = [random.randint(-10, 10) for _ in range(NI * NI)]
    Q_data   = [random.randint(-10, 10) for _ in range(NI * NI)]
    AP_data  = [random.randint(-10, 10) for _ in range(NI * NI)]
    # Inicializamos APA y P a 0 ya que son de salida
    APA_data = [0 for _ in range(NI * NI)]
    P_data   = [0 for _ in range(NI * NI)]

    # Copias de seguridad para la CPU
    A_cpy   = A_data.copy()
    Q_cpy   = Q_data.copy()
    AP_cpy  = AP_data.copy()
    APA_cpy = APA_data.copy()
    P_cpy   = P_data.copy()

    # Configurar memoria y ejecutar kernel
    load_addrs = configMemory(A_data, Q_data, AP_data, APA_data, P_data, NI)
    runKernel(load_addrs, max_it=200000, printVal=0)

    # Cálculo de la dirección base y límite de 'P' (resultado final)
    first_addr_A   = first_addr
    first_addr_Q   = first_addr_A   + (NI * NI * 4)
    first_addr_AP  = first_addr_Q   + (NI * NI * 4)
    first_addr_APA = first_addr_AP  + (NI * NI * 4)
    first_addr_P   = first_addr_APA + (NI * NI * 4)
    end_addr_P     = first_addr_P   + (NI * NI * 4)

    # Obtener resultado del CGRA desde la región de 'P'
    result = getResult(first_addr_P, end_addr_P, NI, NI)

    # Obtener resultado de referencia en CPU
    expected_res = kalman_2_cpu(A_cpy, Q_cpy, AP_cpy, APA_cpy, P_cpy, NI)

    # Comprobar diferencias
    errors = 0
    for i in range(len(expected_res)):
        if expected_res[i] != result[i]:
            errors += 1

    if errors > 0:
        print(f"Err: {errors}")
        print("Expected: ")
        printAsMatrix(expected_res, NI, NI)
        print("CGRA: ")
        printAsMatrix(result, NI, NI)
    else:
        print("OK")

if __name__ == "__main__":
    main()