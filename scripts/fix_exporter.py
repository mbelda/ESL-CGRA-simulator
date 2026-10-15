import csv
import re
import sys

def parse_and_flatten(infile, outfile):
    out_lines = []

    with open(infile, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row or not any(row):
                continue

            # Unir elementos de la fila CSV por si el CSV venía delimitado por comas
            line_content = " ".join(row).strip()

            # Detectar cabecera de tiempo (ej: "Time = 2", "T = 2", "2")
            match = re.match(r"^(?:Time|T)?\s*=?\s*(\d+)$", line_content, re.IGNORECASE)

            if match:
                timestamp = match.group(1)
                out_lines.append(f"T = {timestamp}")
            else:
                # Extraer las instrucciones individuales separadas por 2 o más espacios
                # o por tabulaciones dentro de la fila del CGRA
                instructions = re.split(r"\s{2,}|\t+", line_content)

                for instr in instructions:
                    cleaned_instr = instr.strip()
                    if cleaned_instr:
                        out_lines.append(cleaned_instr)

    # Añadir el marcador final habitual del formato SAT
    if not out_lines or out_lines[-1] != "T = 0":
        out_lines.append("T = 0")

    # Escribir el resultado final
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"[+] Archivo adaptado correctamente a formato SAT en: {outfile}")

def main():
    if len(sys.argv) < 3:
        print("Uso: python3 fix_exporter.py <input.csv> <output.sat>")
        return

    parse_and_flatten(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()