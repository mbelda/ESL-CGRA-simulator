#!/usr/bin/env python3
import sys
import tempfile
import os
import re
import argparse
import csv
import shutil
from collections import defaultdict

parser = argparse.ArgumentParser(description="Procesar instrucciones en CSV.")
parser.add_argument("input", help="CSV a procesar (se sobrescribe por defecto)")
parser.add_argument("--dry-run", action="store_true", help="No sobrescribir, mostrar cambios")
parser.add_argument("--arg-val", type=str, default="4", help="Valor para sustituir argX (por defecto: 4)")
args = parser.parse_args()

input_path = args.input
arg_replacement = args.arg_val

# Regex
#jump_re = re.compile(r'\bJUMP\s*0\s*,?\s*(\d+)\b', flags=re.IGNORECASE)
arg_re = re.compile(r'(?i)arg(\d+)')
sito_fpt_re = re.compile(r'\b(SITOFP|FPTOSI)\b', flags=re.IGNORECASE)

# Leer CSV completo
with open(input_path, newline='', encoding="utf-8") as f:
    reader = list(csv.reader(f))

if not reader:
    print("CSV vacío.")
    sys.exit(0)

n_cols = max(len(row) for row in reader)

for row in reader:
    while len(row) < n_cols:
        row.append("")

# Lista de argX encontrados por columna
args_found = defaultdict(list)

changes = []
out_rows = []

for row_idx, row in enumerate(reader):
    new_row = []

    for col_idx, cell in enumerate(row):

        original_cell = cell
        current_X = None

        stripped = cell.strip()

        # Detectar bloque si la celda es solo un número
        if stripped.isdigit():
            current_X = int(stripped)
            new_row.append(cell)
            continue

        # ---- Sustituir JUMP 0, m ----
        def replace_jump(match):
            m = int(match.group(1))
            if current_X is not None and m == current_X + 1:
                return "NOP"
            else:
                return f"BEQ R0, R0, {m}"

        # new_cell = jump_re.sub(replace_jump, cell)
        new_cell = cell

        # ---- Detectar argX (con repeticiones) ----
        for match in arg_re.finditer(new_cell):
            x_val = int(match.group(1))
            args_found[col_idx].append(x_val)

        # ---- Sustituir argX ----
        new_cell2 = arg_re.sub(arg_replacement, new_cell)

        # ---- Sustituir SITOFP/FPTOSI por NOP ----
        new_cell3 = sito_fpt_re.sub("NOP", new_cell2)

        if new_cell3 != original_cell:
            changes.append((row_idx + 1, col_idx + 1, original_cell, new_cell3))

        new_row.append(new_cell3)

    out_rows.append(new_row)

# ---- Mostrar lista de argX encontrados ----
print("\nArgumentos encontrados por columna:")
for col in range(n_cols):
    print(f"Columna {col}: {args_found[col]}")

# ---- Dry run ----
if args.dry_run:
    if not changes:
        print("\nNo se detectarían cambios.")
    else:
        print(f"\nSe detectarían {len(changes)} cambios:")
        for r, c, old, new in changes:
            print(f"Fila {r}, Col {c}:")
            print(f"  - Antes: {old}")
            print(f"  - Después: {new}")
    sys.exit(0)

# ---- Escribir CSV modificado ----
tmp_filename = "tmp_processed.csv"  # se crea en el directorio actual
try:
    # Crear el CSV temporal en .
    with open(tmp_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    
    # Copiar al path original para sobreescribir
    shutil.copy(tmp_filename, input_path)
    
    # Borrar el temporal
    os.remove(tmp_filename)
    
    print(f"\nArchivo '{input_path}' procesado correctamente. ({len(changes)} cambios)")
except Exception as e:
    try:
        os.remove(tmp_filename)
    except Exception:
        pass
    print("Error al escribir el fichero:", e)
    sys.exit(1)