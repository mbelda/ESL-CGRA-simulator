import csv
import os
import sys
from typing import List, Tuple

# Importamos la función de parseo y las constantes del script principal
try:
    from reg_usage import (
        ALL_REGS,
        N_COLS,
        N_ROWS,
        TOTAL_REGS_PER_PE,
        extract_regs_from_token,
        parse_csv,
    )
except ImportError:
    print(
        "Error: No se encontró 'reg_usage.py' en la misma carpeta."
    )
    sys.exit(1)


def calculate_global_avg_coverage(file_path: str) -> float:
    """Calcula el % de cobertura media de registros en la malla para un fichero CSV dado."""
    try:
        instr_blocks = parse_csv(file_path)
    except Exception as e:
        print(f"  [!] Error al leer '{file_path}': {e}")
        return None

    if not instr_blocks:
        return None

    # Estructura para registrar los registros únicos usados por cada PE
    pe_used_regs = {
        (r, c): set() for r in range(N_ROWS) for c in range(N_COLS)
    }

    # Recorrer todas las instrucciones y acumular los registros detectados
    for block in instr_blocks:
        for r in range(N_ROWS):
            for c in range(N_COLS):
                instr_str = block[r][c]
                regs = extract_regs_from_token(instr_str)
                pe_used_regs[(r, c)].update(regs)

    # Calcular el % por PE y promediar toda la malla (16 PEs)
    total_pes = N_ROWS * N_COLS
    sum_percentages = 0.0

    for r in range(N_ROWS):
        for c in range(N_COLS):
            used_count = len(pe_used_regs[(r, c)])
            pct = (used_count / TOTAL_REGS_PER_PE) * 100
            sum_percentages += pct

    return sum_percentages / total_pes


def process_directory_recursively(
    target_dir: str, output_csv_path: str
) -> List[Tuple[str, str, float]]:
    """Recorre de forma recursiva una carpeta buscando archivos .csv y procesándolos."""
    results = []

    # Normalizar rutas relativas/absolutas
    target_dir = os.path.abspath(target_dir)
    output_csv_abs = os.path.abspath(output_csv_path)

    print(f"Buscando archivos .csv recursivamente en: {target_dir}\n")

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)

                # Ignorar el propio fichero de salida en caso de ser guardado dentro de la misma carpeta
                if os.path.abspath(file_path) == output_csv_abs:
                    continue

                avg_coverage = calculate_global_avg_coverage(file_path)

                if avg_coverage is not None:
                    # Guardamos la ruta absoluta o relativa respecto al punto de inicio
                    rel_dir = os.path.relpath(root, target_dir)
                    results.append((rel_dir, file, round(avg_coverage, 2)))
                    print(f"  [✓] {os.path.join(rel_dir, file)} -> {avg_coverage:.2f}%")

    return results


def save_results_to_csv(
    results: List[Tuple[str, str, float]], output_file: str
):
    """Guarda la lista de resultados en un archivo CSV."""
    headers = [
        "Ruta al fichero",
        "Nombre del fichero",
        "% Cobertura Media Global",
    ]

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rel_dir, filename, coverage in results:
            writer.writerow([rel_dir, filename, coverage])

    print(f"\nResultados guardados con éxito en: {output_file}")


if __name__ == "__main__":
    # Primer argumento: carpeta a analizar (por defecto la carpeta actual '.')
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "."

    # Segundo argumento opcional: nombre del archivo CSV de salida
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "resumen_registros.csv"

    if not os.path.isdir(input_folder):
        print(f"Error: La ruta '{input_folder}' no existe o no es un directorio.")
        sys.exit(1)

    # Procesar recursivamente
    results = process_directory_recursively(input_folder, output_csv)

    # Guardar en CSV si se encontraron archivos
    if results:
        save_results_to_csv(results, output_csv)
    else:
        print("No se encontraron ficheros CSV válidos para procesar.")