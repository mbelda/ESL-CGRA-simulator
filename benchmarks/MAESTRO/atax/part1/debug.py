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

def simulate_atax_p1_cgra(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    # Carga de datos correspondientes a ATAX
    data = np.load(npz_path)
    A = data["A"].copy()
    x = data["x"]
    M = int(data["M"])
    N = int(data["N"])

    # Creación y apertura del fichero de salida dinámico para la Parte 1
    output_filename = f"debug_atax_p1_{M}_{N}_out.log"
    f_out = open(output_filename, "w")

    A_mat = A.reshape((M, N))
    tmp_vec = np.zeros(M, dtype=np.int32)
    instr_cnt = 1

    # Avanza en bloques de 16 elementos de 'i' para cubrir toda la matriz si M > 16.
    for i_base in range(0, M, 16):
        
        # El acumulador local (acc) vive en los registros de la matriz 4x4 de PEs
        acc_local = [[0]*4 for _ in range(4)]

        # Recorrido secuencial o por pasos en el eje J para alimentar los PEs
        for j_base in range(0, N, 4):
            reg_matrix = [[0]*4 for _ in range(4)]

            # ----------------------------------------------------
            # 1. LOAD MATRIZ A[i][j] (Mapeo por filas en el espacio)
            # ----------------------------------------------------
            # El PE(r,c) procesa la fila i = i_base + r * 4 + c
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + (r * 4) + c
                    j = j_base 
                    if i < M and j < N:
                        val_matrix[r][c] = int(A_mat[i, j])
                        op_matrix[r][c] = f"LWI R0 (A[{i}][{j}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 2. LOAD VECTOR x[j] (Broadcast global a todos los PEs)
            # ----------------------------------------------------
            # Todos los PEs reciben simultáneamente el mismo elemento x[j]
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            j_current = j_base
            if j_current < N:
                x_val = int(x[j_current])
                for r in range(4):
                    for c in range(4):
                        val_matrix[r][c] = x_val
                        op_matrix[r][c] = f"LD R6 (x[{j_current}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 3. MULTIPLICACIÓN Y ACUMULACIÓN EN EL ESPACIO
            # ----------------------------------------------------
            accum_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + (r * 4) + c
                    if i < M:
                        # Cada PE acumula sus 4 productos usando el mismo vector x[j] coordinado
                        for step_j in range(4):
                            j = j_base + step_j
                            if j < N:
                                acc_local[r][c] += int(A_mat[i, j]) * int(x[j])
                        
                        accum_matrix[r][c] = acc_local[r][c]
                        op_matrix[r][c] = f"MAC R2 R0 R6 (acc_{i})"
            print_cgra_block(f_out, instr_cnt, accum_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

        # --------------------------------------------------------
        # 4. ESCRITURA FINAL DE TMP[i] (Distribución espacial real)
        # --------------------------------------------------------
        val_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                i = i_base + (r * 4) + c
                if i < M:
                    tmp_vec[i] = acc_local[r][c]
                    val_matrix[r][c] = tmp_vec[i]
                    op_matrix[r][c] = f"SW R2 OUT (tmp[{i}])"
        print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

    f_out.close()
    print(f"Traza de debug (ATAX P1 Broadcast) guardada en: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Simulador de Debug Espacial ATAX P1 con Broadcast de X")
    parser.add_argument("--npz", type=str, required=True, help="Ruta al fichero de datos .npz")
    args = parser.parse_args()

    simulate_atax_p1_cgra(args.npz)

if __name__ == "__main__":
    main()