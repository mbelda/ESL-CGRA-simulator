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

def configMemory(tmp_data, A_data, B_data, C_data, D_data, NI, NJ, NK, NL):
    kernel_clear_memory(".", version=version)
    
    # Cálculo de direcciones iniciales (4 bytes por entero)
    first_addr_tmp = first_addr
    first_addr_A   = first_addr_tmp + (NI * NJ * 4)
    first_addr_B   = first_addr_A   + (NI * NK * 4)
    first_addr_C   = first_addr_B   + (NK * NJ * 4)
    first_addr_D   = first_addr_C   + (NJ * NL * 4)

    config_cols = [
        [first_addr_C, first_addr_D, first_addr_D],
        [first_addr_tmp, first_addr_D, first_addr_tmp, first_addr_D],
        [first_addr_B],
        [first_addr_A, first_addr_tmp]
    ]
    
    load_addrs = []
    current_addr = 0
    
    for config_vals in config_cols:
        load_addrs.append(current_addr)
        kernel_add_memory_region(".", current_addr, config_vals, version=version)
        current_addr += len(config_vals) * 4
    
    data_regions = [
        (first_addr_A, A_data),
        (first_addr_B, B_data),
        (first_addr_C, C_data),
        (first_addr_D, D_data)
    ]
    
    for addr, data in data_regions:
        kernel_add_memory_region(".", addr, data, version=version)
    
    return load_addrs

def runKernel(load_addrs, max_it=1000, printVal=0):
    # Cambiado kernel_name por "." para buscar instructions en el directorio actual
    run(".", pr=["ROUT", "R0", "R1", "INST"], load_addrs=load_addrs, version=version, limit=max_it, printVal=printVal)

def getResult(start_addr, end_addr, rows, cols):
    result = [0 for _ in range(rows * cols)]
    # Lectura directa del archivo local en la carpeta del benchmark
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

def twomm_cpu(A_data, B_data, C_data, D_data, NI, NJ, NK, NL, alpha, beta):
    # tmp = alpha * A * B  (NI x NJ)
    tmp = [0 for _ in range(NI * NJ)]
    for i in range(NI):
        for j in range(NJ):
            sum_val = 0
            for k in range(NK):
                sum_val += A_data[i * NK + k] * B_data[k * NJ + j]
            tmp[i * NJ + j] = alpha * sum_val

    # D = tmp * C + beta * D  (NI x NL)
    expected_res = [0 for _ in range(NI * NL)]
    for i in range(NI):
        for l in range(NL):
            sum_val = 0
            for j in range(NJ):
                sum_val += tmp[i * NJ + j] * C_data[j * NL + l]
            expected_res[i * NL + l] = sum_val + beta * D_data[i * NL + l]
    return expected_res

# -----------------------------------------------------------------------------
# Flujo Principal
# -----------------------------------------------------------------------------
def main():
    print(f"[-] Ejecutando kernel: {kernel_name}")

    # Dimensiones estándar
    NI = dim
    NJ = dim
    NK = dim
    NL = dim

    alpha = 1  # Nota: Revisa si tu ensamblador CGRA usa alpha/beta constantes o simplificadas
    beta = 1

    # Generar datos aleatorios según las dimensiones del header C:
    # A[NI][NK], B[NK][NJ], C[NJ][NL], D[NI][NL], tmp[NI][NJ]
    tmp_data = [0 for _ in range(NI * NJ)]
    A_data   = [random.randint(-10, 10) for _ in range(NI * NK)]
    B_data   = [random.randint(-10, 10) for _ in range(NK * NJ)]
    C_data   = [random.randint(-10, 10) for _ in range(NJ * NL)]
    D_data   = [random.randint(-10, 10) for _ in range(NI * NL)]

    A_cpy = A_data.copy()
    B_cpy = B_data.copy()
    C_cpy = C_data.copy()
    D_cpy = D_data.copy()

    # Configurar memoria y ejecutar kernel
    load_addrs = configMemory(tmp_data, A_data, B_data, C_data, D_data, NI, NJ, NK, NL)
    runKernel(load_addrs, max_it=200000, printVal=0)

    # Cálculo de la dirección base de D para recuperar el resultado
    first_addr_tmp = first_addr
    first_addr_A   = first_addr_tmp + (NI * NJ * 4)
    first_addr_B   = first_addr_A   + (NI * NK * 4)
    first_addr_C   = first_addr_B   + (NK * NJ * 4)
    first_addr_D   = first_addr_C   + (NJ * NL * 4)
    end_addr_D     = first_addr_D   + (NI * NL * 4)

    # Obtener resultado del CGRA (Asumiendo salida en D)
    result = getResult(first_addr_D, end_addr_D, NI, NL)

    # Obtener resultado de referencia en CPU
    expected_res = twomm_cpu(A_cpy, B_cpy, C_cpy, D_cpy, NI, NJ, NK, NL, alpha, beta)

    # Comprobar si son iguales
    errors = 0
    for i in range(len(expected_res)):
        if expected_res[i] != result[i]:
            errors += 1

    if errors > 0:
        print(f"Err: {errors}")
        print("Expected: ")
        printAsMatrix(expected_res, NI, NL)
        print("CGRA: ")
        printAsMatrix(result, NI, NL)
    else:
        print("OK")

if __name__ == "__main__":
    main()