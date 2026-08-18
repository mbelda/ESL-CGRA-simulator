#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
import os
import re
import shutil
import sys
import tempfile


def parse_header_args(header_str):
    """Extrae los nombres de los parámetros de la firma de una función C/C++."""
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


def generate_config_memory_snippet(args_found, n_cols, header_args):
    """Genera la estructura config_cols formateada con las listas por columna."""
    lines = []
    lines.append("\n" + "=" * 50)
    lines.append("# Snippet generado para configMemory")
    lines.append("=" * 50)
    lines.append("config_cols = [")

    for col in range(n_cols):
        var_names = []
        for arg_idx in args_found[col]:
            if 0 <= arg_idx < len(header_args):
                var_names.append(f"first_addr_{header_args[arg_idx]}")
            else:
                var_names.append(f"first_addr_arg{arg_idx}")

        val_list = ", ".join(var_names)
        comma = "," if col < n_cols - 1 else ""
        lines.append(f"    [{val_list}]{comma}")

    lines.append("]")
    lines.append("=" * 50 + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Procesar instrucciones en CSV y generar snippet para configMemory."
    )
    parser.add_argument(
        "input", help="CSV a procesar (se sobrescribe por defecto)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No sobrescribir, mostrar cambios",
    )
    parser.add_argument(
        "--arg-val",
        type=str,
        default="4",
        help="Valor para sustituir argX (por defecto: 4)",
    )
    parser.add_argument(
        "--header",
        type=str,
        default="",
        help="Header/firma de la función C",
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
        print("CSV vacío.")
        sys.exit(0)

    n_cols = max(len(row) for row in reader)

    for row in reader:
        while len(row) < n_cols:
            row.append("")

    args_found = defaultdict(list)
    changes = []
    out_rows = []

    for row_idx, row in enumerate(reader):
        new_row = []

        for col_idx, cell in enumerate(row):
            original_cell = cell
            stripped = cell.strip()

            if stripped.isdigit():
                new_row.append(cell)
                continue

            for match in arg_re.finditer(cell):
                x_val = int(match.group(1))
                args_found[col_idx].append(x_val)

            new_cell = arg_re.sub(arg_replacement, cell)
            new_cell_final = sito_fpt_re.sub("NOP", new_cell)

            if new_cell_final != original_cell:
                changes.append(
                    (row_idx + 1, col_idx + 1, original_cell, new_cell_final)
                )

            new_row.append(new_cell_final)

        out_rows.append(new_row)

    print("\nArgumentos encontrados por columna:")
    for col in range(n_cols):
        print(f"Columna {col}: {args_found[col]}")

    header_args = parse_header_args(args.header) if args.header else []
    if args.header:
        print(f"\nParámetros extraídos del header: {header_args}")

    print(generate_config_memory_snippet(args_found, n_cols, header_args))

    if args.dry_run:
        if not changes:
            print("No se detectarían cambios en el CSV.")
        else:
            print(f"Se detectarían {len(changes)} cambios:")
            for r, c, old, new in changes:
                print(f"Fila {r}, Col {c}:")
                print(f"  - Antes: {old}")
                print(f"  - Después: {new}")
        sys.exit(0)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=input_dir, delete=False
        ) as tmp_file:
            writer = csv.writer(tmp_file)
            writer.writerows(out_rows)
            tmp_filename = tmp_file.name

        shutil.move(tmp_filename, input_path)
        print(
            f"Archivo '{input_path}' procesado correctamente. ({len(changes)} cambios)"
        )
    except Exception as e:
        if "tmp_filename" in locals() and os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except Exception:
                pass
        print("Error al escribir el fichero:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()