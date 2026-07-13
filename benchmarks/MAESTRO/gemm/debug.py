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

def simulate_gemm_cgra(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    # Carga de datos correspondientes a GEMM
    data = np.load(npz_path)
    A = data["A"].copy()
    B = data["B"].copy()
    C = data["C"].copy()
    
    # Extraemos dimensiones y escalares estilo Polybench
    NI = int(data["rowsA"]) if "rowsA" in data else int(data["NI"])
    NK = int(data["colsA"]) if "colsA" in data else int(data["NK"])
    NJ = int(data["colsB"]) if "colsB" in data else int(data["NJ"])
    
    alpha = int(data["alpha"]) if "alpha" in data else 1
    beta = int(data["beta"]) if "beta" in data else 1

    # Reconstrucción de la geometría bidimensional
    A_mat = A.reshape((NI, NK))
    B_mat = B.reshape((NK, NJ))
    C_mat = C.reshape((NI, NJ))
    C_out = np.zeros((NI, NJ), dtype=np.int32)

    # Nombre dinámico basado en las dimensiones reales
    suffix = f"{NI}" if (NI == NK == NJ) else f"{NI}_{NK}_{NJ}"
    output_filename = f"debug_gemm_{suffix}_out.log"
    f_out = open(output_filename, "w")

    instr_cnt = 1
    reg_matrix = [[0]*4 for _ in range(4)]

    # ------------------------------------------------------------------
    # NUEVO ORDEN DE TILING: Primero recorre bloques por columnas (j), 
    # y luego avanza a la siguiente fila de bloques (i)
    # ------------------------------------------------------------------
    
    for i_base in range(0, NI, 4):
        for j_base in range(0, NJ, 4):
            
            # Inicializamos el acumulador local interno del bloque PE de la CGRA en 0
            acc_local = [[0]*4 for _ in range(4)]

            # ----------------------------------------------------
            # 1. BUCLE DE REDUCCIÓN EN K (Multiplicación y acumulación por alpha)
            # ----------------------------------------------------
            for k in range(NK):
                
                # --- LOAD ELEMENTOS DE MATRIZ A ---
                val_matrix = [[0]*4 for _ in range(4)]
                op_matrix = [["NOP"]*4 for _ in range(4)]
                for r in range(4):
                    for c in range(4):
                        i = i_base + r
                        if i < NI and k < NK:
                            val_matrix[r][c] = int(A_mat[i, k])
                            op_matrix[r][c] = f"LWI R0 (A[{i}][{k}])"
                print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
                instr_cnt += 1

                # --- LOAD ELEMENTOS DE MATRIZ B ---
                val_matrix = [[0]*4 for _ in range(4)]
                op_matrix = [["NOP"]*4 for _ in range(4)]
                for r in range(4):
                    for c in range(4):
                        j = j_base + c
                        if k < NK and j < NJ:
                            val_matrix[r][c] = int(B_mat[k, j])
                            op_matrix[r][c] = f"LWI R1 (B[{k}][{j}])"
                print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
                instr_cnt += 1

                # --- MULTIPLICACIÓN Y ACUMULACIÓN INTERNA (acc += A * B * alpha) ---
                val_matrix = [[0]*4 for _ in range(4)]
                op_matrix = [["NOP"]*4 for _ in range(4)]
                for r in range(4):
                    for c in range(4):
                        i = i_base + r
                        j = j_base + c
                        if i < NI and j < NJ:
                            acc_local[r][c] += int(A_mat[i, k]) * int(B_mat[k, j]) * alpha
                            val_matrix[r][c] = acc_local[r][c]
                            op_matrix[r][c] = f"MAC R2 R0 R1 * alpha"
                print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
                instr_cnt += 1

            # ----------------------------------------------------
            # 2. CARGA DE C, ESCALADO POR BETA Y SUMA FINAL
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < NI and j < NJ:
                        # Se calcula la suma final integrando beta * C[i][j] al acumulado local
                        final_val = acc_local[r][c] + (beta * int(C_mat[i, j]))
                        C_out[i, j] = final_val
                        val_matrix[r][c] = final_val
                        op_matrix[r][c] = f"ADD R2 R2 (C*{beta})"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # ----------------------------------------------------
            # 3. ESCRITURA EN MEMORIA DEL RESULTADO FINAL DE C
            # ----------------------------------------------------
            val_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i = i_base + r
                    j = j_base + c
                    if i < NI and j < NJ:
                        val_matrix[r][c] = int(C_out[i, j])
                        op_matrix[r][c] = f"SW R2 OUT (C[{i}][{j}])"
            print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

    f_out.close()
    print(f"Traza de debug (GEMM espacial {NI}x{NK}x{NJ}) guardada en: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Simulador de Debug Espacial GEMM con escalado alpha/beta")
    parser.add_argument("--npz", type=str, required=True, help="Ruta al fichero de datos .npz")
    args = parser.parse_args()

    simulate_gemm_cgra(args.npz)

if __name__ == "__main__":
    main()