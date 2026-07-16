#!/usr/bin/env python3

import os
import sys
import csv
import argparse
from pathlib import Path

# ------------------------------------------------------------------
#  DYNAMIC ROOT PATH INJECTION (3 levels up for MAESTRO layout)
# ------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

# repo/benchmarks/MAESTRO/3loops/ -> repo/
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

parser = argparse.ArgumentParser(description="Script para ejecutar CGRA 3-loops kernel con dimensiones dinámicas.")
parser.add_argument('--channels', type=int, required=True, help='Número de canales (C)')
parser.add_argument('--kh', type=int, required=True, help='Kernel Height (KH)')
parser.add_argument('--kw', type=int, required=True, help='Kernel Width (KW)')
parser.add_argument('--ih', type=int, required=True, help='Input Height (IH)')
parser.add_argument('--iw', type=int, required=True, help='Input Width (IW)')

args = parser.parse_args()

channels = args.channels
kh = args.kh
kw = args.kw
ih = args.ih
iw = args.iw
tam_canal_input = ih * iw

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
log_filename = f"cgra_3loops_c{channels}_kh{kh}_kw{kw}_iw{iw}_output.log"

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
def configMemory(input_lider, weights, channels, kh, kw, ih, iw):
    kernel_clear_memory(kernel_name, version=version)
    
    # Tamaños en bytes (enteros de 4 bytes)
    size_input = len(input_lider) * 4
    size_weights = len(weights) * 4

    # Direcciones base físicas
    first_addr_in = first_addr
    first_addr_w = first_addr_in + size_input
    
    # Espacio reservado para el entero resultante justo tras los pesos
    store_address = first_addr_w + size_weights  

    # Constantes físicas de desplazamiento en memoria
    size_ch_im = 4 * ih * iw
    size_ch_f  = 4 * kh * kw

    # Direcciones de canales específicos de Imagen (Im) y Filtros (F)
    addr_Im_0  = first_addr_in + (0 * size_ch_im)
    addr_Im_6  = first_addr_in + (6 * size_ch_im)
    addr_Im_9  = first_addr_in + (9 * size_ch_im)
    addr_Im_15 = first_addr_in + (15 * size_ch_im)

    addr_F_2   = first_addr_w + (2 * size_ch_f)
    addr_F_4   = first_addr_w + (4 * size_ch_f)
    addr_F_11  = first_addr_w + (11 * size_ch_f)
    addr_F_13  = first_addr_w + (13 * size_ch_f)

    # Iteradores de bucle (N - 1 para loops de hardware CGRA)
    loopCIt  = int(channels/16) - 1
    loopKHIt = kh - 1
    loopKWIt = kw - 1

    # --------------------------------------------------------------------------
    # MAPEO ESTRUCTURADO DE REGISTROS DE CONFIGURACIÓN DEL CGRA (COLUMNAS 0 A 3)
    # --------------------------------------------------------------------------
    # Columna 0               Columna 1             Columna 2             Columna 3
    # --------------------------------------------------------------------------
    # &Im[0][0][0]            4*IH*IW               &F[2][0][0]           4*KH*KW
    # &F[4][0][0]             4*KH*KW               &Im[6][0][0]          4*IH*IW
    # 4*KH*KW                 &Im[9][0][0]          4*IH*IW               &F[11][0][0]
    # 4*IH*IW                 &F[13][0][0]          4*KH*KW               &Im[15][0][0]
    # --------------------------------------------------------------------------
    # IH                      IW                    KH                    KW
    # KH                      KW                    loopKHIt              loopKWIt
    # KW                      IH                    IW                    KH
    # loopCIt                 KH                    KW                    IW     
    # --------------------------------------------------------------------------
    # 0                       store_address         0                     0
    # --------------------------------------------------------------------------
    config_vals = [[] for i in range(CGRA_N_COLS)]

    config_vals[0] = [addr_Im_0, addr_F_4, size_ch_f, size_ch_im, ih, kh, kw, loopCIt, 0]
    config_vals[1] = [size_ch_im, size_ch_f, addr_Im_9, addr_F_13, iw, kw, ih, kh, store_address]
    config_vals[2] = [addr_F_2, addr_Im_6, size_ch_im, size_ch_f, kh, loopKHIt, iw, kw, 0]
    config_vals[3] = [size_ch_f, size_ch_im, addr_F_11, addr_Im_15, kw, loopKWIt, kh, iw, 0]
    # Carga automática del bloque de configuración en el simulador
    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS - 1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i]) * 4
            
    # Mapeo de datos de entrada en la memoria simulada
    kernel_add_memory_region(kernel_name, first_addr_in, input_lider, version=version)
    kernel_add_memory_region(kernel_name, first_addr_w, weights, version=version)

    # Vector con las direcciones físicas de escritura de salida por cada PE
    store_addresses = [0, store_address, 0, 0]

    return addr_config_loads, store_addresses


def runKernel(load_addrs=None, store_addrs=None, max_it=1000, pr=["ROUT","INST"], printVal=1):
    sys.stdout.mute_terminal = True
    try:
        run(kernel_name, pr=pr, load_addrs=load_addrs, store_addrs=store_addrs, version=version, limit=max_it, printVal=printVal)
    finally:
        sys.stdout.mute_terminal = False


def getResult(first_addr, length=1):
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
# Formato del sufijo usado en la generación: c{C}_kh{KH}_kw{KW}_iw{IW}
suffix = f"c{channels}_kh{kh}_kw{kw}_iw{iw}"
data_path = os.path.join(kernel_name, "data", f"data_{suffix}.npz")

if not os.path.exists(data_path):
    print(f"Error: No se encontró el archivo de datos {data_path}")
    sys.exit(1)

data = np.load(data_path)

input_lider = data["input_lider"]
weights = data["weights"]
expected_sum = int(data["expected_sum"])  

# --- CÁLCULO PREVIO DE DIRECCIONES PARA MOSTRAR AL INICIO ---
size_input = len(input_lider) * 4
size_weights = len(weights) * 4

first_addr_in = first_addr
first_addr_w = first_addr_in + size_input
# Dirección reservada para almacenar el resultado final de la simulación
expected_store_addr = first_addr_w + size_weights

print("====================================================")
print("             CGRA MEMORY MAPPING DETAILS            ")
print("====================================================")
print(f"Base Address (first_addr) : {first_addr}")
print(f"Input Lider Base Address  : {first_addr_in} (Size: {len(input_lider)} ints)")
print(f"Weights Base Address      : {first_addr_w} (Size: {len(weights)} ints)")
print(f"Result Sum Store Address  : {expected_store_addr}")
print("====================================================\n")

print(f"Testing CGRA 3-Loops with configuration: C={channels}, KH={kh}, KW={kw}, IH={ih}, IW={iw}")
print(f"Full log execution details will write silently to: {log_filename}\n")

# Configurar memoria pasando las entradas y dimensiones
load_addrs, store_addresses = configMemory(input_lider, weights, channels, kh, kw, ih, iw)

# Obtenemos la dirección de almacenamiento correspondiente a la columna 1
store_address_col1 = store_addresses[1]

# Ejecución del Kernel simulado
runKernel(load_addrs=load_addrs, store_addrs=store_addresses, max_it=2000000, pr=["ROUT", "R2", "INST"], printVal=1)

# Recuperar el resultado acumulado (1 entero) de la dirección de almacenamiento de la Columna 1
cgra_output = getResult(store_address_col1, length=1)
obtained_sum = cgra_output[0]

# Verificación de resultados
print("Check CGRA output result:")
if obtained_sum == expected_sum:
    print(f"OK (Obtained: {obtained_sum} | Expected: {expected_sum})")
else:
    print(f"FAIL -> Expected: {expected_sum} | CGRA Obtained: {obtained_sum}")