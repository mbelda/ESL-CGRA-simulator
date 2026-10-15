#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
import json
import os
import re
import shutil
import sys
import tempfile


def parse_header_args(header_str):
    header_str = " ".join(header_str.split())
    match = re.search(r"\((.*)\)", header_str)
    if not match:
        return []

    params_str = match.group(1).strip()
    if not params_str or params_str == "void":
        return []

    args = []
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        param_clean = re.sub(r"\[.*?\]", "", param).strip()
        tokens = re.findall(r"\b[a-zA-Z_]\w*\b", param_clean)
        if tokens:
            args.append(tokens[-1])
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Procesar instrucciones en CSV y generar snippet para configMemory."
    )
    parser.add_argument("input", help="CSV a procesar")
    parser.add_argument(
        "--arg-val", type=str, default="4", help="Valor para sustituir argX"
    )
    parser.add_argument(
        "--header", type=str, default="", help="Header/firma de la función C"
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Ruta para guardar config_cols en formato JSON",
    )
    args = parser.parse_args()

    input_path = args.input
    arg_replacement = args.arg_val
    input_dir = os.path.dirname(input_path) or "."

    arg_re = re.compile(r"\barg(\d+)\b", flags=re.IGNORECASE)
    sito_fpt_re = re.compile(r"\b(SITOFP|FPTOSI)\b", flags=re.IGNORECASE)

    try:
        with open(input_path, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
    except Exception as e:
        print(f"Error al abrir '{input_path}': {e}")
        sys.exit(1)

    if not reader:
        sys.exit(0)

    n_cols = max(len(row) for row in reader)
    args_found = defaultdict(list)
    out_rows = []

    for row_idx, row in enumerate(reader):
        new_row = []
        for col_idx, cell in enumerate(row):
            stripped = cell.strip()
            if stripped.isdigit():
                new_row.append(cell)
                continue

            for match in arg_re.finditer(cell):
                x_val = int(match.group(1))
                args_found[col_idx].append(x_val)

            new_cell = arg_re.sub(arg_replacement, cell)
            new_cell_final = sito_fpt_re.sub("NOP", new_cell)
            new_row.append(new_cell_final)

        out_rows.append(new_row)

    header_args = parse_header_args(args.header) if args.header else []

    # Construcción de la estructura de configuración mapeada
    config_cols_mapped = []
    for col in range(n_cols):
        col_list = []
        for arg_idx in args_found[col]:
            if 0 <= arg_idx < len(header_args):
                col_list.append(header_args[arg_idx])
            else:
                col_list.append(f"arg{arg_idx}")
        config_cols_mapped.append(col_list)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as jf:
            json.dump(
                {"n_cols": n_cols, "config_cols": config_cols_mapped}, jf
            )

    # Reescritura del CSV
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=input_dir, delete=False
    ) as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerows(out_rows)
        tmp_filename = tmp_file.name

    shutil.move(tmp_filename, input_path)


if __name__ == "__main__":
    main()