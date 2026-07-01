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
            
            # 1. PASO: DIRECCIONES DE MEMORIA DE LA MATRIZ A
            addr_matrix = [[0]*4 for _ in range(4)]
            reg_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        addr_matrix[r][c] = base_A + (i * N + j) * 4
                        op_matrix[r][c] = f"SADD ROUT R2 ZERO"
            
            print_cgra_block(f_out, instr_cnt, addr_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 2. PASO: LOADS DEL VALOR BASE DE A[i, j]
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        val_matrix[r][c] = int(A_mat[i, j])
                        op_matrix[r][c] = f"LWI R2 ROUT (A[{i}][{j}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 3. PASO: CÓMPUTO DEL PRIMER TÉRMINO (u1[i] * v1[j])
            term1_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        val_u1 = int(u1[i])
                        val_v1 = int(v1[j])
                        term1_matrix[r][c] = val_u1 * val_v1
                        op_matrix[r][c] = f"MUL R1 (u1[{i}]*v1[{j}])"
            print_cgra_block(f_out, instr_cnt, term1_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 4. PASO: CÓMPUTO Y ACUMULACIÓN DEL SEGUNDO TÉRMINO (u2[i] * v2[j])
            accum_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < N and j < N:
                        val_u2 = int(u2[i])
                        val_v2 = int(v2[j])
                        delta = term1_matrix[r][c] + (val_u2 * val_v2)
                        A_mat[i, j] += delta
                        accum_matrix[r][c] = int(A_mat[i, j])
                        op_matrix[r][c] = f"MAC R3 (u2[{i}]*v2[{j}])"
            print_cgra_block(f_out, instr_cnt, accum_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 5. PASO: ESCRITURA EN MEMORIA DEL RESULTADO ACTUALIZADO
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