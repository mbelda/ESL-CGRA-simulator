#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse

def print_cgra_block(file_out, instr_id, matrix_data, reg_data, op_strings):
    """Escribe la salida simulando el formato de volcado de debug en el archivo."""
    file_out.write(f"Instr =  {instr_id} ( {instr_id} )\n")
    for row in range(4):
        m_str = "[" + ", ".join(f"{matrix_data[row][col]:>5}" for col in range(4)) + "]"
        r_str = "[" + ", ".join(f"{reg_data[row][col]:>4}" for col in range(4)) + "]"
        o_str = "[" + ", ".join(f"{op_strings[row][col]:<22}" for col in range(4)) + "]"
        file_out.write(f"{m_str}    {r_str}    {o_str}\n")
    file_out.write("Aprox cycles this pc: 1\n")
    file_out.write("-------\n")

def simulate_atax_p2_cgra(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    data = np.load(npz_path)
    A = data["A"].copy()
    tmp = data["tmp"]
    M = int(data["M"])
    N = int(data["N"])

    output_filename = f"debug_atax_p2_{M}_{N}_out.log"
    f_out = open(output_filename, "w")

    A_mat = A.reshape((M, N))
    y_vec = np.zeros(N, dtype=np.int32)
    instr_cnt = 1

    # Paralelismo: j_base avanza de 16 en 16 distribuyéndose por columnas en el CGRA (PE_r_c -> j)
    for j_base in range(0, N, 16):
        acc_local = [[0]*4 for _ in range(4)]

        # Recorrido secuencial e iterativo por CADA fila 'i' de la matriz
        for i in range(M):
            reg_matrix = [[0]*4 for _ in range(4)]

            # 1. LOAD MATRIZ A[i][j] (Se carga exactamente la fila 'i' actual)
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    j = j_base + (r * 4) + c
                    if i < M and j < N:
                        val_matrix[r][c] = int(A_mat[i, j])
                        op_matrix[r][c] = f"LWI R0 (A[{i}][{j}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 2. LOAD VECTOR tmp[i] (Broadcast exacto del elemento 'i' correspondiente)
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            if i < M:
                tmp_val = int(tmp[i])
                for r in range(4):
                    for c in range(4):
                        val_matrix[r][c] = tmp_val
                        op_matrix[r][c] = f"LD R6 (tmp[{i}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 3. MAC ESPACIAL (Se opera estrictamente el ciclo actual: sin bucles fantasma)
            accum_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    j = j_base + (r * 4) + c
                    if j < N and i < M:
                        # Acumulación limpia ciclo a ciclo
                        acc_local[r][c] += int(A_mat[i, j]) * int(tmp[i])
                        accum_matrix[r][c] = acc_local[r][c]
                        op_matrix[r][c] = f"MAC R2 R0 R6 (acc_A*tmp)"
            print_cgra_block(f_out, instr_cnt, accum_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

        # --------------------------------------------------------
        # 4. LOAD DEL VALOR ACTUAL DE y[j] (Al salir de TODAS las filas i)
        # --------------------------------------------------------
        val_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        reg_matrix = [[0]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                j = j_base + (r * 4) + c
                if j < N:
                    val_matrix[r][c] = int(y_vec[j])
                    op_matrix[r][c] = f"LWI R4 (y[{j}]_old)"
        print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

        # --------------------------------------------------------
        # 5. SUMA FINAL: acc_local = acc_local + y[j]_old
        # --------------------------------------------------------
        accum_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                j = j_base + (r * 4) + c
                if j < N:
                    acc_local[r][c] += int(y_vec[j])
                    accum_matrix[r][c] = acc_local[r][c]
                    op_matrix[r][c] = f"ADD R2 R2 R4 (y[{j}]_new)"
        print_cgra_block(f_out, instr_cnt, accum_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

        # --------------------------------------------------------
        # 6. ESCRITURA EN MEMORIA DEL RESULTADO FINAL
        # --------------------------------------------------------
        val_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                j = j_base + (r * 4) + c
                if j < N:
                    y_vec[j] = acc_local[r][c]
                    val_matrix[r][c] = y_vec[j]
                    op_matrix[r][c] = f"SW R2 OUT (y[{j}])"
        print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

    f_out.close()
    print(f"Traza de debug corregida (coherente ciclo a ciclo) en: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Simulador de Debug Espacial ATAX P2 Acumulativo Síncrono")
    parser.add_argument("--npz", type=str, required=True, help="Ruta al fichero de datos .npz")
    args = parser.parse_args()
    simulate_atax_p2_cgra(args.npz)

if __name__ == "__main__":
    main()