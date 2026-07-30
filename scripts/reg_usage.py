import csv
import sys
from typing import Dict, List, Set

# Configuración del CGRA
N_ROWS = 4
N_COLS = 4
INSTR_SIZE = N_ROWS + 1  # 1 fila de timestamp/ID + 4 filas de operaciones PE

# Conjunto total de registros por PE
ALL_REGS: Set[str] = {"R0", "R1", "R2", "R3", "ROUT"}
TOTAL_REGS_PER_PE = len(ALL_REGS)  # 5 registros


def parse_csv(file_path: str) -> List[List[List[str]]]:
    """Lee el CSV y extrae las matrices de instrucciones 4x4."""
    raw_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                raw_lines.append(row)

    instr_blocks = []
    num_instructions = len(raw_lines) // INSTR_SIZE

    for i in range(num_instructions):
        block = raw_lines[i * INSTR_SIZE : (i + 1) * INSTR_SIZE]
        ops_matrix = [block[r + 1] for r in range(N_ROWS)]
        instr_blocks.append(ops_matrix)

    return instr_blocks


def extract_regs_from_token(token: str) -> Set[str]:
    """Extrae qué registros válidos de la lista ALL_REGS aparecen en una instrucción."""
    clean_instr = token.replace(",", " ").strip()
    tokens = clean_instr.split()

    if not tokens or tokens[0] == "NOP":
        return set()

    found_regs = set()
    for t in tokens[1:]:  # Ignorar el mnemónico de la operación (tokens[0])
        if t in ALL_REGS:
            found_regs.add(t)

    return found_regs


def analyze_register_coverage(file_path: str):
    instr_blocks = parse_csv(file_path)

    if not instr_blocks:
        print("El archivo CSV no contiene datos válidos.")
        return

    # Estructura para registrar los registros únicos usados por cada PE
    pe_used_regs: Dict[tuple, Set[str]] = {
        (r, c): set() for r in range(N_ROWS) for c in range(N_COLS)
    }

    # Recorrer todas las instrucciones y acumular los registros detectados
    for block in instr_blocks:
        for r in range(N_ROWS):
            for c in range(N_COLS):
                instr_str = block[r][c]
                regs = extract_regs_from_token(instr_str)
                pe_used_regs[(r, c)].update(regs)

    # Imprimir Informe
    print("=" * 75)
    print(f" ANÁLISIS DE COBERTURA DE REGISTROS POR PE (Catálogo Total: 5)")
    print(f" Archivo: {file_path}")
    print("=" * 75)

    print("\n[1] MATRIZ DE % DE USO DE REGISTROS POR PE:")
    print("    (% = registros únicos utilizados / 5)")
    print("-" * 65)

    header = "      " + "".join([f"  Col {c}   " for c in range(N_COLS)])
    print(header)

    percentages = []

    for r in range(N_ROWS):
        row_str = f"Row {r} "
        for c in range(N_COLS):
            used_count = len(pe_used_regs[(r, c)])
            pct = (used_count / TOTAL_REGS_PER_PE) * 100
            percentages.append(pct)
            row_str += f" | {pct:5.1f}% "
        print(row_str + " |")

    print("\n[2] DETALLE DE REGISTROS USADOS POR CADA PE:")
    print("-" * 65)

    for r in range(N_ROWS):
        for c in range(N_COLS):
            used = pe_used_regs[(r, c)]
            used_str = (
                ", ".join(sorted(used)) if used else "Ninguno (solo NOPs)"
            )
            pct = (len(used) / TOTAL_REGS_PER_PE) * 100
            print(
                f" - PE [{r},{c}]: {len(used)}/5 registros ({pct:5.1f}%) -> [{used_str}]"
            )

    avg_coverage = sum(percentages) / len(percentages)

    print("\n[3] RESUMEN GLOBAL:")
    print("-" * 65)
    print(
        f" * COBERTURA MEDIA DE REGISTROS EN LA MALLA: {avg_coverage:.2f}% por PE"
    )
    print("=" * 75)


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "instructions.csv"
    analyze_register_coverage(csv_file)