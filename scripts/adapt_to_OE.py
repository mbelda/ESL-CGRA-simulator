#!/usr/bin/env python3
import sys
import tempfile
import os
import re
import argparse

parser = argparse.ArgumentParser(description="Procesar instrucciones en fichero.")
parser.add_argument("input", help="Fichero a procesar (se sobrescribe por defecto)")
parser.add_argument("--dry-run", action="store_true", help="No sobrescribir, mostrar cambios")
parser.add_argument("--arg-val", type=str, default="4", help="Valor para sustituir argX (por defecto: 4)")
args = parser.parse_args()

input_path = args.input
arg_replacement = args.arg_val

# Regex para "JUMP 0, m"
jump_re = re.compile(r'\bJUMP\s*0\s*,\s*(\d+)\b', flags=re.IGNORECASE)
# Regex más robusta para "argX" (captura arg seguido de dígitos, sin depender de \b)
arg_re = re.compile(r'(?i)arg(\d+)')  # captura el número si hace falta

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

out_lines = []
current_X = None
changes = []  # para dry-run: (lineno, old, new)

for idx, line in enumerate(lines, start=1):
    stripped = line.strip()

    # Detectar número de bloque
    if stripped.isdigit():
        current_X = int(stripped)
        out_lines.append(line)
        continue

    if current_X is None:
        out_lines.append(line)
        continue

    # ---- Sustituir JUMP 0, m ----
    def replace_jump(match):
        m_str = match.group(1)
        try:
            m = int(m_str)
        except ValueError:
            return match.group(0)
        if m == current_X + 1:
            return "NOP"
        else:
            return f"BEQ R0, R0, {m}"

    new_line = jump_re.sub(replace_jump, line)

    # ---- Sustituir argX → arg_val (solo la parte argX) ----
    # Usamos una función para preservar la posible capitalización o formato
    def replace_arg(match):
        # match.group(0) es por ejemplo "arg1" o "ARG12"
        # Sustituimos por el valor proporcionado (por defecto "4")
        return arg_replacement

    new_line2 = arg_re.sub(replace_arg, new_line)

    if new_line2 != line:
        changes.append((idx, line.rstrip("\n"), new_line2.rstrip("\n")))

    out_lines.append(new_line2)

# Si dry-run, mostramos los cambios y no sobrescribimos
if args.dry_run:
    if not changes:
        print("No se detectaron cambios.")
    else:
        print(f"Se detectarían {len(changes)} cambios:")
        for lineno, old, new in changes:
            print(f"{lineno}:")
            print(f"  - Antes: {old}")
            print(f"  - Después: {new}")
    sys.exit(0)

# Escribir resultado a temporal y reemplazar
tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
try:
    tmp.writelines(out_lines)
    tmp.close()
    os.replace(tmp.name, input_path)
    print(f"Archivo '{input_path}' procesado y sobrescrito correctamente. ({len(changes)} cambios)")
except Exception as e:
    try:
        os.remove(tmp.name)
    except Exception:
        pass
    print("Error al escribir el fichero:", e)
    sys.exit(1)
