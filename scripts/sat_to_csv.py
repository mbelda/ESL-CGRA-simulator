import argparse
import csv
import math
import os
import sys


def convert(infile, outfile, version=""):
    """Convierte la salida de SAT-MapIt en un CSV compatible con el simulador."""

    # Inserción limpia de la versión antes de la extensión del archivo
    if version:
        base, ext = os.path.splitext(outfile)
        outfile = f"{base}_{version}{ext}"

    # Leer el archivo de entrada
    try:
        with open(infile, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error al leer el archivo de entrada: {e}")
        sys.exit(1)

    reading_conf = False
    last_time_stamp = -1
    current_time_stamp = 0
    current_config = []
    conf_set = []

    for line in lines:
        if line.startswith("T ="):
            current_time_stamp = int(line.split(" ")[-1].strip())

            # Si el timestamp disminuye, hemos terminado
            if last_time_stamp != -1 and current_time_stamp < last_time_stamp:
                if current_config:
                    conf_set.append(current_config)
                break

            # Guardamos la configuración previa antes de iniciar una nueva
            if current_config:
                conf_set.append(current_config)

            reading_conf = True
            last_time_stamp = current_time_stamp
            current_config = [line]
            continue

        if reading_conf:
            current_config.append(line)

    # Añadir la última configuración procesada
    if reading_conf and current_config:
        conf_set.append(current_config)

    if not conf_set:
        print("No se encontraron configuraciones en el archivo.")
        return None

    # Infección de filas y columnas (malla cuadrada)
    n_nodes = len(conf_set[0][1:])
    n_cols = int(math.sqrt(n_nodes))
    n_rows = n_cols
    print(f"Malla detectada: {n_cols}x{n_rows} ({n_nodes} nodos)")

    # Crear directorio de salida si no existe
    out_dir = os.path.dirname(outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Escribir el archivo CSV de salida
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for conf in conf_set:
            time_line = conf[0]
            time_val = int(time_line.split(" ")[-1].strip())

            instrs = conf[1:]

            # Escribir timestamp
            writer.writerow([time_val])

            # Organizar e instruir las filas de la malla
            rows = [
                [instrs[(n_cols * r) + c].strip() for c in range(n_cols)]
                for r in range(n_rows)
            ]

            for r in rows:
                writer.writerow(r)

    return outfile


def main():
    parser = argparse.ArgumentParser(
        description="Convierte la salida de SAT-MapIt a un archivo CSV para el simulador."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Ruta al archivo de entrada (ej: ./datos/sat_output.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Ruta al archivo de salida (ej: ./resultados/sim_input.csv)",
    )
    parser.add_argument(
        "-v",
        "--version",
        default="",
        help="Etiqueta opcional de versión para añadir al nombre de salida",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: El archivo de entrada '{args.input}' no existe.")
        sys.exit(1)

    final_outfile = convert(args.input, args.output, args.version)

    if final_outfile:
        print(f" Proceso completado. Archivo guardado en: '{final_outfile}'")


if __name__ == "__main__":
    main()