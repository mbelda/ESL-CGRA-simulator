import csv
import sys

def analizar_csv(nombre_fichero):
    total_instr = 0
    total_nops = 0
    max_ciclo = -1

    with open(nombre_fichero, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        i = 0
        while i < len(reader):
            fila = reader[i]
            if fila and fila[0].isdigit():
                ciclo_num = int(fila[0])
                max_ciclo = max(max_ciclo, ciclo_num)
                # Leemos las 4 filas siguientes (bloque 4x4)
                for j in range(1, 5):
                    if i + j < len(reader):
                        fila_instrucciones = reader[i + j]
                        for instr in fila_instrucciones:
                            instr = instr.strip()
                            if instr == "":
                                continue
                            total_instr += 1
                            if instr.upper() == "NOP":
                                total_nops += 1
                i += 5
            else:
                i += 1

    total_ciclos = max_ciclo + 1
    porcentaje_nops = 100 * total_nops / total_instr if total_instr > 0 else 0
    return total_instr, total_nops, porcentaje_nops, total_ciclos


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <fichero.csv>")
        sys.exit(1)

    archivo = sys.argv[1]
    total, nops, porcentaje, ciclos = analizar_csv(archivo)

    print(f"Instrucciones ejecutadas (no vacías): {total}")
    print(f"NOPs: {nops}")
    print(f"Porcentaje de NOPs: {porcentaje:.2f}%")
    print(f"Número total de ciclos: {ciclos}")
