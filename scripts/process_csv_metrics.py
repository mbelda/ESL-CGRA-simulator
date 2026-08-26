import sys
import re
import argparse
import pandas as pd
from pathlib import Path

# Orden personalizado de los tests según la imagen
BENCHMARK_ORDER = [
    "mmul_base",
    "mmul_relu",
    "mmul_batch",
    "2mm",
    "3mm",
    "gemm",
    "PCA",
    "kalman_1",
    "kalman_2",
    "qmmul"
]

def clean_and_sort_csv(csv_path: str, overwrite: bool = False):
    input_file = Path(csv_path).resolve()

    if not input_file.exists():
        print(f"Error: No se encuentra el archivo {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file)

    # 1. Eliminar filas con FAILED_EXECUTION
    df = df[df["Status"] != "FAILED_EXECUTION"]

    # 2. Eliminar duplicados
    df = df.drop_duplicates()

    # Variables auxiliares para la ordenación
    
    # Tamaño CGRA en formato numérico (ej. "3x3" -> 3)
    df["cgra_size_num"] = df["Grid_Size"].apply(lambda x: int(str(x).split("x")[0]) if "x" in str(x) else 0)

    # Normalizar nombre del benchmark (ej. "PCA/v2" -> "PCA")
    df["clean_bench_name"] = df["Benchmark_Type"].apply(lambda x: str(x).split("/")[0])
    
    # Asignar índice de orden personalizado
    bench_mapping = {name: i for i, name in enumerate(BENCHMARK_ORDER)}
    df["bench_order"] = df["clean_bench_name"].map(lambda x: bench_mapping.get(x, 99))

    # Tamaño de datos desde la columna Version (ej. "_3_IJK60" -> 60)
    def extract_data_size(version_str):
        match = re.search(r"IJK(\d+)", str(version_str))
        return int(match.group(1)) if match else 0

    df["data_size"] = df["Version"].apply(extract_data_size)

    # 3. Ordenar: CGRA (3x3, 4x4, 5x5) -> Tests (imagen) -> Tamaño datos (60, 24)
    df = df.sort_values(
        by=["cgra_size_num", "bench_order", "data_size"],
        ascending=[True, True, False]
    )

    # Limpiar columnas auxiliares
    df = df.drop(columns=["cgra_size_num", "clean_bench_name", "bench_order", "data_size"])

    # Definir ruta de salida (Crea un fichero nuevo por defecto en el mismo directorio)
    output_file = input_file if overwrite else input_file.parent / f"{input_file.stem}_clean{input_file.suffix}"

    # Guardar CSV procesado
    df.to_csv(output_file, index=False)
    print(f"✔ CSV procesado y guardado con éxito en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa y ordena el archivo CSV de métricas.")
    parser.add_argument("csv_path", type=str, help="Ruta al archivo CSV original.")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribir el archivo original en lugar de crear uno nuevo.")

    args = parser.parse_args()
    clean_and_sort_csv(args.csv_path, overwrite=args.overwrite)