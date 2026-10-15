#!/usr/bin/env python3
import argparse
from contextlib import redirect_stdout
import csv
import json
import os
import random
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuración de rutas e importaciones dinámicas
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # /ESL-CGRA-simulator

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

kernel_name = str(SCRIPT_DIR.relative_to(PROJECT_ROOT))

from cgra import *
from kernels import *

first_addr = 128


def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols : (i + 1) * cols])


def configMemory(
    inputX_data,
    inputY_data,
    output_data,
    NI,
    NJ,
    NK,
    version_tag,
    config_cols_raw,
):
    # Al estar posicionados en csv_dir, pasamos "." como directorio
    kernel_clear_memory(".", version=version_tag)

    first_addr_inputX = first_addr
    first_addr_inputY = first_addr_inputX + (NI * NK * 4)
    first_addr_output = first_addr_inputY + (NK * NJ * 4)

    addr_map = {
        "inputX": first_addr_inputX,
        "inputY": first_addr_inputY,
        "output": first_addr_output,
    }

    # Mapeo dinámico de nombres de variables a sus direcciones reales
    config_cols = []
    for col_vars in config_cols_raw:
        col_addrs = [
            addr_map[var_name]
            for var_name in col_vars
            if var_name in addr_map
        ]
        config_cols.append(col_addrs)

    load_addrs = []
    current_addr = 0

    for config_vals in config_cols:
        load_addrs.append(current_addr)
        kernel_add_memory_region(
            ".", current_addr, config_vals, version=version_tag
        )
        current_addr += len(config_vals) * 4

    data_regions = [
        (first_addr_inputX, inputX_data),
        (first_addr_inputY, inputY_data),
        (first_addr_output, output_data),
    ]

    for addr, data in data_regions:
        kernel_add_memory_region(".", addr, data, version=version_tag)

    return load_addrs


def runKernel(load_addrs, version_tag, max_it=20000000, printVal=1):
    run(
        ".",
        pr=["ROUT", "R0", "R1", "R2", "INST"],
        load_addrs=load_addrs,
        version=version_tag,
        limit=max_it,
        printVal=printVal,
    )


def getResult(start_addr, end_addr, rows, cols, version_tag):
    result = [0 for _ in range(rows * cols)]
    csv_file_path = Path(f"memory_out{version_tag}.csv")

    if not csv_file_path.exists():
        return result

    with open(csv_file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:
            try:
                addr = int(row[0])
                if start_addr <= addr < end_addr:
                    result[int((addr - start_addr) / 4)] = int(row[1])
            except (ValueError, IndexError):
                pass
    return result


def mmul_relu_cpu(inputX_data, inputY_data, NI, NJ, NK):
    expected_res = [0 for _ in range(NI * NJ)]
    for i in range(NI):
        for j in range(NJ):
            sum_val = 0
            for k in range(NK):
                sum_val += inputX_data[i * NK + k] * inputY_data[k * NJ + j]
            expected_res[i * NJ + j] = sum_val

    for i in range(NI * NJ):
        if expected_res[i] < 0:
            expected_res[i] = 0

    return expected_res


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutor de benchmarks genérico"
    )
    parser.add_argument(
        "--version-tag", required=True, help="Etiqueta de versión/ejemplo"
    )
    parser.add_argument(
        "--config-json", required=True, help="JSON con el config_cols"
    )
    parser.add_argument(
        "--benchmark-dir", required=True, help="Directorio del benchmark (raíz del modo)"
    )
    parser.add_argument("--ni", type=int, required=True, help="Dimensión NI")
    parser.add_argument("--nj", type=int, required=True, help="Dimensión NJ")
    parser.add_argument("--nk", type=int, required=True, help="Dimensión NK")

    args = parser.parse_args()

    benchmark_dir = os.path.abspath(args.benchmark_dir)
    csv_dir = os.path.join(benchmark_dir, "csv")
    config_json_path = os.path.abspath(args.config_json)

    # Nos posicionamos directamente en csv/ para resolver las instrucciones y ficheros auxiliares
    os.chdir(csv_dir)

    version_tag = f"_{args.version_tag}"
    log_file_path = os.path.join(benchmark_dir, f"execution{version_tag}.log")

    with open(config_json_path, "r") as jf:
        cfg_data = json.load(jf)
        config_cols_raw = cfg_data["config_cols"]

    NI, NJ, NK = args.ni, args.nj, args.nk

    inputX_data = [random.randint(-10, 10) for _ in range(NI * NK)]
    inputY_data = [random.randint(-10, 10) for _ in range(NK * NJ)]
    output_data = [0 for _ in range(NI * NJ)]

    inputX_cpy = inputX_data.copy()
    inputY_cpy = inputY_data.copy()

    errors = 0
    with open(log_file_path, "w", encoding="utf-8") as log_f:
        with redirect_stdout(log_f):
            print(f"[-] Ejecutando kernel: {kernel_name} ({version_tag})")

            load_addrs = configMemory(
                inputX_data,
                inputY_data,
                output_data,
                NI,
                NJ,
                NK,
                version_tag,
                config_cols_raw,
            )
            runKernel(load_addrs, version_tag, max_it=200000000, printVal=1)

            first_addr_inputX = first_addr
            first_addr_inputY = first_addr_inputX + (NI * NK * 4)
            first_addr_output = first_addr_inputY + (NK * NJ * 4)
            end_addr_output = first_addr_output + (NI * NJ * 4)

            result = getResult(
                first_addr_output, end_addr_output, NI, NJ, version_tag
            )
            expected_res = mmul_relu_cpu(inputX_cpy, inputY_cpy, NI, NJ, NK)

            errors = sum(
                1 for e, r in zip(expected_res, result) if e != r
            )

            if errors > 0:
                print(f"Err: {errors}")
                print("Expected:")
                printAsMatrix(expected_res, NI, NJ)
                print("CGRA:")
                printAsMatrix(result, NI, NJ)
            else:
                print("OK")

    if errors > 0:
        print(f"❌ ERROR en {version_tag}: {errors} fallos detectados.")
    else:
        print(f"✅ OK en {version_tag}: Ejecución correcta.")


if __name__ == "__main__":
    main()