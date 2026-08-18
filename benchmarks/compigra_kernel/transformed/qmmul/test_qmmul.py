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

def configMemory(A_data, B_data, bias_data, scale_data, C_data, NI, NJ, NK):
    kernel_clear_memory(".", version=version)
    
    # Cálculo de direcciones iniciales (4 bytes por entero)
    first_addr_A     = first_addr
    first_addr_B     = first_addr_A     + (NI * NK * 4)
    first_addr_bias  = first_addr_B     + (NK * NJ * 4)
    first_addr_scale = first_addr_bias  + (NI * 4)
    first_addr_C     = first_addr_scale + (NJ * 4)

    # Configuración de columnas provista
    config_cols = [
        [first_addr_scale],
        [first_addr_A, first_addr_bias],
        [first_addr_B, first_addr_C],
        []
    ]
    
    load_addrs = []
    current_addr = 0
    
    for config_vals in config_cols:
        load_addrs.append(current_addr)
        kernel_add_memory_region(".", current_addr, config_vals, version=version)
        current_addr += len(config_vals) * 4
    
    # Carga de datos de entrada y salida
    data_regions = [
        (first_addr_A, A_data),
        (first_addr_B, B_data),
        (first_addr_bias, bias_data),
        (first_addr_scale, scale_data),
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

def qmmul_cpu(A_data, B_data, bias_data, scale_data, NI, NJ, NK):
    C = [0 for _ in range(NI * NJ)]
    
    for i in range(NI):
        for j in range(NJ):
            acc = 0
            for k in range(NK):
                a = A_data[i * NK + k] + bias_data[i]
                b = B_data[k * NJ + j] * scale_data[j]
                acc += a * b
            C[i * NJ + j] = acc
            
    return C

# -----------------------------------------------------------------------------
# Flujo Principal
# -----------------------------------------------------------------------------
def main():
    print(f"[-] Ejecutando kernel: {kernel_name}")

    NI = dim
    NJ = dim
    NK = dim

    # Generar datos aleatorios de entrada
    A_data     = [random.randint(-10, 10) for _ in range(NI * NK)]
    B_data     = [random.randint(-10, 10) for _ in range(NK * NJ)]
    bias_data  = [random.randint(-10, 10) for _ in range(NI)]
    scale_data = [random.randint(-10, 10) for _ in range(NJ)]
    
    # Inicializar búfer de salida a cero
    C_data     = [0 for _ in range(NI * NJ)]

    # Copias de seguridad para la CPU
    A_cpy     = A_data.copy()
    B_cpy     = B_data.copy()
    bias_cpy  = bias_data.copy()
    scale_cpy = scale_data.copy()

    # Configurar memoria y ejecutar kernel
    load_addrs = configMemory(A_data, B_data, bias_data, scale_data, C_data, NI, NJ, NK)
    runKernel(load_addrs, max_it=200000, printVal=0)

    # Cálculo de la dirección base y límite de 'C' (resultado final)
    first_addr_A     = first_addr
    first_addr_B     = first_addr_A     + (NI * NK * 4)
    first_addr_bias  = first_addr_B     + (NK * NJ * 4)
    first_addr_scale = first_addr_bias  + (NI * 4)
    first_addr_C     = first_addr_scale + (NJ * 4)
    end_addr_C       = first_addr_C     + (NI * NJ * 4)

    # Obtener resultado del CGRA desde la región de 'C'
    result = getResult(first_addr_C, end_addr_C, NI, NJ)

    # Obtener resultado de referencia en CPU
    expected_res = qmmul_cpu(A_cpy, B_cpy, bias_cpy, scale_cpy, NI, NJ, NK)

    # Comprobar diferencias
    errors = 0
    for i in range(len(expected_res)):
        if expected_res[i] != result[i]:
            errors += 1

    if errors > 0:
        print(f"Err: {errors}")
        print("Expected: ")
        printAsMatrix(expected_res, NI, NJ)
        print("CGRA: ")
        printAsMatrix(result, NI, NJ)
    else:
        print("OK")

if __name__ == "__main__":
    main()