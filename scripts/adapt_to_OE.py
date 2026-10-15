#!/usr/bin/env python3
import argparse
import math
import os
import re
import sys
import tempfile

# Expresiones regulares
jump_re = re.compile(r"\bJUMP\s*0\s*,?\s*(\d+)\b", flags=re.IGNORECASE)
arg_re = re.compile(r"(?i)arg(\d+)")
block_re = re.compile(r"^(?:T\s*=\s*)?(\d+)$", flags=re.IGNORECASE)


def process_file(input_path, arg_replacement, dry_run=False):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error al abrir el archivo: {e}")
        sys.exit(1)

    out_lines = []
    changes = []

    jump_nop_count = 0
    jump_beq_count = 0
    lwd_count = 0

    jump_details = []  # Para guardar los detalles de los JUMP -> BEQ

    current_X = None
    instr_index = 0

    N_COLS = 4
    col_args = [[] for _ in range(N_COLS)]

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            out_lines.append(line)
            continue

        # Detectar cabecera de bloque (ej: "T = 0" o "0")
        match_block = block_re.match(stripped)
        if match_block:
            current_X = int(match_block.group(1))
            instr_index = 0  # Reiniciar índice de instrucción dentro del nuevo bloque
            out_lines.append(line)
            continue

        if current_X is None:
            out_lines.append(line)
            continue

        # Calcular coordenadas del PE dentro de la malla 4x4
        row = instr_index // N_COLS
        col = instr_index % N_COLS
        pe_id = instr_index  # ID de PE del 0 al 15

        # 1. ---- Procesar JUMP ----
        def replace_jump(match):
            nonlocal jump_nop_count, jump_beq_count
            m_str = match.group(1)
            try:
                target_block = int(m_str)
            except ValueError:
                return match.group(0)

            # Si salta al bloque inmediatamente siguiente (current_X + 1)
            if target_block == current_X + 1:
                jump_nop_count += 1
                return "NOP"
            else:
                # Salto a otro bloque (no adyacente)
                jump_beq_count += 1
                jump_details.append(
                    f"Bloque T={current_X} | PE[{row}][{col}] (Nodo {pe_id}) -> Salta a Bloque {target_block}"
                )
                return f"BEQ R0, R0, {target_block}"

        new_line = jump_re.sub(replace_jump, line)

        # 2. ---- Detectar y reemplazar ARGx ----
        arg_matches = arg_re.findall(new_line)
        if arg_matches:
            for arg_num in arg_matches:
                col_args[col].append(f"ARG{arg_num}")
                lwd_count += 1

            new_line = arg_re.sub(arg_replacement, new_line)

        if new_line != line:
            changes.append((idx, line.rstrip("\n"), new_line.rstrip("\n")))

        out_lines.append(new_line)
        instr_index += 1

    # ---- Resumen por Terminal ----
    print("\n" + "=" * 60)
    print("                RESUMEN DE PROCESAMIENTO")
    print("=" * 60)
    print(
        f"• JUMP -> NOP (bloque siguiente): {jump_nop_count}"
    )
    print(
        f"• JUMP -> BEQ (salto lejano)    : {jump_beq_count}"
    )
    print(
        f"• Instrucciones LWD/ARG cambiadas: {lwd_count}"
    )
    print("-" * 60)

    if jump_details:
        print(" Detalle de JUMPs a otros bloques (BEQ):")
        for detail in jump_details:
            print(f"   - {detail}")
        print("-" * 60)

    print(" Configuración de ARGx detectados por Columna (CGRA 4x4):")
    for c in range(N_COLS):
        args_str = ", ".join(col_args[c]) if col_args[c] else "ninguno"
        print(f"   Columna {c}: [{args_str}]")
    print("=" * 60 + "\n")

    if dry_run:
        if not changes:
            print("No se detectaron cambios en el archivo.")
        else:
            print(f"Se realizarían {len(changes)} cambios:")
            for lineno, old, new in changes:
                print(f"  Línea {lineno}: {old.strip()}  ==>  {new.strip()}")
        return

    # ---- Sobrescribir el archivo ----
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    try:
        tmp.writelines(out_lines)
        tmp.close()
        os.replace(tmp.name, input_path)
        print(f"✓ Archivo '{input_path}' procesado y guardado.")
    except Exception as e:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        print("Error al escribir el archivo:", e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Procesa instrucciones de SAT-MapIt con análisis detallado de saltos y PEs."
    )
    parser.add_argument("input", help="Fichero a procesar")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No sobrescribir el archivo, solo mostrar cambios",
    )
    parser.add_argument(
        "--arg-val",
        type=str,
        default="4",
        help="Valor para sustituir argX (por defecto: 4)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: El archivo '{args.input}' no existe.")
        sys.exit(1)

    process_file(args.input, args.arg_val, args.dry_run)


if __name__ == "__main__":
    main()