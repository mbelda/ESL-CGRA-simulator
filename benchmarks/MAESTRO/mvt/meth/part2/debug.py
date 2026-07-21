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

def simulate_mvt_p2_cgra(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    # Carga de datos correspondientes a MVT (compatibles con el script unificado)
    data = np.load(npz_path)
    A = data["A"].copy()
    y2 = data["y2"]
    
    # Soporta tanto "x2_init" del full script como variantes
    x2_init = data["x2_init"] if "x2_init" in data else data["x2"]
    N = int(data["N"])

    # Creación y apertura del fichero de salida dinámico para MVT Parte 2
    output_filename = f"debug_mvt_p2_{N}_out.log"
    f_out = open(output_filename, "w")

    A_mat = A.reshape((N, N))
    x2_vec = np.copy(x2_init)
    instr_cnt = 1

    # Avanza en bloques de 16 elementos de 'i' (columnas en el espacio para P2)
    for i_base in range(0, N, 16):
        
        # ----------------------------------------------------
        # 0. PRE-LOAD: Cargar el estado inicial de x2[i] en los PEs
        # ----------------------------------------------------
        acc_local = [[0]*4 for _ in range(4)]
        val_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        reg_matrix = [[0]*4 for _ in range(4)]
        
        for r in range(4):
            for c in range(4):
                i = i_base + (r * 4) + c
                if i < N:
                    acc_local[r][c] = int(x2_vec[i])
                    val_matrix[r][c] = acc_local[r][c]
                    op_matrix[r][c] = f"LW R2 (x2[{i}] init)"
        print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

        # Recorrido secuencial o por pasos en el eje J (filas de A) para alimentar los PEs
        for j_base in range(0, N, 4):
            reg_matrix = [[0]*4 for _ in range(4)]

            # ----------------------------------------------------
            # 1. LOAD MATRIZ A[j][i] (Mapeo por columnas/transpuesto)
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + (r * 4) + c
                    j = j_base 
                    if i < N and j < N:
                        # MVT P2 utiliza A[j][i]
                        val_matrix[r][c] = int(A_mat[j, i])
                        op_matrix[r][c] = f"LWI R0 (A[{j}][{i}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 2. LOAD VECTOR y2[j] (Broadcast global a todos los PEs)
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            j_current = j_base
            if j_current < N:
                y2_val = int(y2[j_current])
                for r in range(4):
                    for c in range(4):
                        val_matrix[r][c] = y2_val
                        op_matrix[r][c] = f"LD R6 (y2[{j_current}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 3. MULTIPLICACIÓN Y ACUMULACIÓN SOBRE EL VALOR PREVIO
            # ----------------------------------------------------
            accum_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + (r * 4) + c
                    if i < N:
                        for step_j in range(4):
                            j = j_base + step_j
                            if j < N:
                                # Acumulación del producto transpuesto
                                acc_local[r][c] += int(A_mat[j, i]) * int(y2[j])
                        
                        accum_matrix[r][c] = acc_local[r][c]
                        op_matrix[r][c] = f"MAC R2 R0 R6 (x2_{i})"
            print_cgra_block(f_out, instr_cnt, accum_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

        # --------------------------------------------------------
        # 4. ESCRITURA FINAL DE LA ACUMULACIÓN DE X2[i]
        # --------------------------------------------------------
        val_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                i = i_base + (r * 4) + c
                if i < N:
                    x2_vec[i] = acc_local[r][c]
                    val_matrix[r][c] = x2_vec[i]
                    op_matrix[r][c] = f"SW R2 OUT (x2[{i}])"
        print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

    f_out.close()
    print(f"Traza de debug (MVT P2 Broadcast) guardada en: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Simulador de Debug Espacial MVT P2 con Broadcast de Y2")
    parser.add_argument("--npz", type=str, required=True, help="Ruta al fichero de datos .npz")
    args = parser.parse_args()

    simulate_mvt_p2_cgra(args.npz)

if __name__ == "__main__":
    main()