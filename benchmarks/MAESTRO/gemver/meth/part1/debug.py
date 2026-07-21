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

def simulate_gemver_p1_cgra(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    # Carga de datos correspondientes a la parte 1
    data = np.load(npz_path)
    A = data["A"].copy()
    u1 = data["u1"]
    v1 = data["v1"]
    u2 = data["u2"]
    v2 = data["v2"]
    N = int(data["NI"] if "NI" in data else np.sqrt(len(A)))

    # Creación y apertura del fichero de salida dinámico: debug_N_out.log
    output_filename = f"debug_{N}_out.log"
    f_out = open(output_filename, "w")

    # Direcciones base (mapeo secuencial de la Parte 1)
    base_A  = 20000
    base_u1 = base_A + (N * N * 4)
    base_v1 = base_u1 + (N * 4)
    base_u2 = base_v1 + (N * 4)
    base_v2 = base_u2 + (N * 4)

    A_mat = A.reshape((N, N))
    instr_cnt = 1

    # Recorremos por filas (i) y columnas (j) mapeando el paralelismo del CGRA
    for i_base in range(0, N, 4):
        for j_base in range(0, N, 4):
            
            # Reutilizaremos matrices de registros vacías para estados intermedios
            reg_matrix = [[0]*4 for _ in range(4)]

            # ----------------------------------------------------
            # 1. LOAD BASE A[i][j]
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        val_matrix[r][c] = int(A_mat[i, j])
                        op_matrix[r][c] = f"LWI R2 (A[{i}][{j}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 2. LOAD u1[i]
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    if i < N:
                        val_matrix[r][c] = int(u1[i])
                        op_matrix[r][c] = f"LD R4 (u1[{i}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 3. LOAD v1[j]
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    j = j_base + c
                    if j < N:
                        val_matrix[r][c] = int(v1[j])
                        op_matrix[r][c] = f"LD R5 (v1[{j}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 4. MULTIPLICACIÓN TÉRMINO 1: u1[i] * v1[j]
            # ----------------------------------------------------
            term1_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        term1_matrix[r][c] = int(u1[i]) * int(v1[j])
                        op_matrix[r][c] = f"MUL R6 R4 R5"
            print_cgra_block(f_out, instr_cnt, term1_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 5. LOAD u2[i]
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    if i < N:
                        val_matrix[r][c] = int(u2[i])
                        op_matrix[r][c] = f"LD R7 (u2[{i}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 6. LOAD v2[j]
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    j = j_base + c
                    if j < N:
                        val_matrix[r][c] = int(v2[j])
                        op_matrix[r][c] = f"LD R8 (v2[{j}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 7. MULTIPLICACIÓN TÉRMINO 2: u2[i] * v2[j]
            # ----------------------------------------------------
            term2_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        term2_matrix[r][c] = int(u2[i]) * int(v2[j])
                        op_matrix[r][c] = f"MUL R9 R7 R8"
            print_cgra_block(f_out, instr_cnt, term2_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 8. SUMAR TÉRMINOS: (u1*v1) + (u2*v2)
            # ----------------------------------------------------
            sum_terms_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        sum_terms_matrix[r][c] = term1_matrix[r][c] + term2_matrix[r][c]
                        op_matrix[r][c] = f"ADD R10 R6 R9"
            print_cgra_block(f_out, instr_cnt, sum_terms_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 9. ACUMULAR SOBRE A: A[i][j] + suma_términos
            # ----------------------------------------------------
            accum_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        A_mat[i, j] += sum_terms_matrix[r][c]
                        accum_matrix[r][c] = int(A_mat[i, j])
                        op_matrix[r][c] = f"ADD R3 R2 R10"
            print_cgra_block(f_out, instr_cnt, accum_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 10. ESCRITURA EN MEMORIA (ST / SW)
            # ----------------------------------------------------
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        op_matrix[r][c] = f"SW R3 OUT (A[{i}][{j}])"
            print_cgra_block(f_out, instr_cnt, accum_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

    f_out.close()
    print(f"Traza de simulación guardada con éxito en: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Simulador del Algoritmo GEMVER P1 al estilo Traza CGRA")
    parser.add_argument("--npz", type=str, required=True, help="Ruta al fichero de datos .npz de la parte 1")
    args = parser.parse_args()

    simulate_gemver_p1_cgra(args.npz)

if __name__ == "__main__":
    main()