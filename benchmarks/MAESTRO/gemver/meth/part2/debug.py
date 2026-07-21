#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse

def print_cgra_block(instr_id, matrix_data, reg_data, op_strings):
    """Formatea la salida simulando el formato de volcado de debug del CGRA."""
    print(f"Instr =  {instr_id} ( {instr_id} )")
    for row in range(4):
        m_str = "[" + ", ".join(f"{matrix_data[row][col]:>5}" for col in range(4)) + "]"
        r_str = "[" + ", ".join(f"{reg_data[row][col]:>4}" for col in range(4)) + "]"
        o_str = "[" + ", ".join(f"{op_strings[row][col]:<22}" for col in range(4)) + "]"
        print(f"{m_str}    {r_str}    {o_str}")
    print("Aprox cycles this pc: 1")
    print("-------")

def simulate_gemver_p2_cgra(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    # Carga de datos
    data = np.load(npz_path)
    A = data["A"]
    x = data["x"].copy()
    y = data["y"]
    z = data["z"]
    beta = int(data["beta"])
    N = int(data["NI"])

    # Direcciones base (asumiendo tu mapeo secuencial)
    base_A = 20000
    base_x = base_A + (N * N * 4)
    base_y = base_x + (N * 4)
    base_z = base_y + (N * 4)

    A_mat = A.reshape((N, N))
    instr_cnt = 1

    # Bucle externo con desenrollado de 16 (bloques de i en pasos de 16)
    for i_block in range(0, N, 16):
        # 1. PASO: DIRECCIONES DE MEMORIA (Loads de direcciones para z y x)
        # Se mapean las i del bloque actual a la cuadrícula 4x4 de PEs
        addr_matrix = [[0]*4 for _ in range(4)]
        reg_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        
        for r in range(4):
            for c in range(4):
                i_idx = i_block + (r * 4 + c)
                if i_idx < N:
                    addr_matrix[r][c] = base_z + (i_idx * 4)
                    op_matrix[r][c] = f"SADD ROUT R2 ZERO"
        
        print_cgra_block(instr_cnt, addr_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

        # 2. PASO: LOADS DE LOS VALORES INICIALES DE Z
        val_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                i_idx = i_block + (r * 4 + c)
                if i_idx < N:
                    val_matrix[r][c] = int(z[i_idx])
                    op_matrix[r][c] = "LWI R2 ROUT"
        print_cgra_block(instr_cnt, val_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

        # Inicializamos los acumuladores locales de la reducción (aux = 0) para cada PE
        aux_accum = [[0]*4 for _ in range(4)]

        # Bucle interno J (Reducción)
        for j in range(N):
            # 3. PASO J: LOAD de A[j][i] (Matriz Transpuesta)
            # Cada PE lee su celda correspondiente a su 'i_idx' actual y al 'j' global
            a_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i_idx = i_block + (r * 4 + c)
                    if i_idx < N:
                        a_matrix[r][c] = int(A_mat[j, i_idx])
                        op_matrix[r][c] = f"LWI R1 ROUT (A[{j}][{i_idx}])"
            print_cgra_block(instr_cnt, a_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 4. PASO J: COMPUTO Y ACUMULACIÓN (aux += beta * A[j][i] * y[j])
            y_val = int(y[j])
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i_idx = i_block + (r * 4 + c)
                    if i_idx < N:
                        prod = beta * a_matrix[r][c] * y_val
                        aux_accum[r][c] += prod
                        # Mostramos el estado intermedio del acumulador en la matriz principal
                        a_matrix[r][c] = aux_accum[r][c]
                        op_matrix[r][c] = f"MAC_ADD R3 (b*{y_val}*A)"
            print_cgra_block(instr_cnt, a_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

        # 5. PASO FINAL DEL BLOQUE: SUMA FINAL Y ESCRITURA (x[i] += aux + z[i])
        res_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                i_idx = i_block + (r * 4 + c)
                if i_idx < N:
                    val_inicial_x = int(x[i_idx])
                    val_z = int(z[i_idx])
                    # x[i] final = x_inicial + aux + z
                    x[i_idx] += aux_accum[r][c] + val_z
                    res_matrix[r][c] = int(x[i_idx])
                    op_matrix[r][c] = f"SW R3 OUT (x[{i_idx}])"
        print_cgra_block(instr_cnt, res_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

def main():
    parser = argparse.ArgumentParser(description="Simulador del Algoritmo GEMVER P2 al estilo Traza CGRA")
    parser.add_argument("--npz", type=str, required=True, help="Ruta al fichero de datos .npz")
    args = parser.parse_args()

    simulate_gemver_p2_cgra(args.npz)

if __name__ == "__main__":
    main()