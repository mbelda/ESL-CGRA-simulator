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

def simulate_gemver_p3_cgra(npz_path):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    # Carga de datos
    data = np.load(npz_path)
    A = data["A"]
    x = data["x"]
    w = data["w"].copy()
    alpha = int(data["alpha"])
    N = int(data["NI"])

    # Direcciones base (Secuencial Parte 3)
    base_A = 20000
    base_x = base_A + (N * N * 4)
    base_w = base_x + (N * 4)

    A_mat = A.reshape((N, N))
    instr_cnt = 1

    # Bucle externo con desenrollado de 16 para i (filas de A / elementos de w)
    for i_block in range(0, N, 16):
        # 1. PASO: DIRECCIONES DE MEMORIA (Loads de direcciones para w e inicialización)
        addr_matrix = [[0]*4 for _ in range(4)]
        reg_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        
        for r in range(4):
            for c in range(4):
                # Siguiendo el mapeo del hardware de la P3: cada PE calcula un w[i]
                i_idx = i_block + (r * 4 + c)
                if i_idx < N:
                    addr_matrix[r][c] = base_w + (i_idx * 4)
                    op_matrix[r][c] = f"SADD ROUT R2 ZERO"
        
        print_cgra_block(instr_cnt, addr_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

        # 2. PASO: LOADS DE LOS VALORES INICIALES DE W
        val_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                i_idx = i_block + (r * 4 + c)
                if i_idx < N:
                    val_matrix[r][c] = int(w[i_idx])
                    op_matrix[r][c] = "LWI R2 ROUT"
        print_cgra_block(instr_cnt, val_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

        # Inicializamos los acumuladores locales de la reducción (aux = 0) para cada PE
        aux_accum = [[0]*4 for _ in range(4)]

        # Bucle interno J (Reducción sobre las columnas de A y elementos de x)
        for j in range(N):
            # 3. PASO J: LOAD de A[i][j] (Lectura directa por fila i)
            a_matrix = [[0]*4 for _ in range(4)]
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i_idx = i_block + (r * 4 + c)
                    if i_idx < N:
                        a_matrix[r][c] = int(A_mat[i_idx, j])
                        op_matrix[r][c] = f"LWI R1 ROUT (A[{i_idx}][{j}])"
            print_cgra_block(instr_cnt, a_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

            # 4. PASO J: COMPUTO Y ACUMULACIÓN (aux += alpha * A[i][j] * x[j])
            x_val = int(x[j])
            op_matrix = [["NOP"]*4 for _ in range(4)]
            for r in range(4):
                for c in range(4):
                    i_idx = i_block + (r * 4 + c)
                    if i_idx < N:
                        prod = alpha * a_matrix[r][c] * x_val
                        aux_accum[r][c] += prod
                        # Mostramos el estado intermedio del acumulador en la matriz principal
                        a_matrix[r][c] = aux_accum[r][c]
                        op_matrix[r][c] = f"MAC_ADD R3 (a*{x_val}*A)"
            print_cgra_block(instr_cnt, a_matrix, reg_matrix, op_matrix)
            instr_cnt += 1

        # 5. PASO FINAL DEL BLOQUE: SUMA FINAL Y ESCRITURA EN W (w[i] = w_inicial[i] + aux)
        res_matrix = [[0]*4 for _ in range(4)]
        op_matrix = [["NOP"]*4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                i_idx = i_block + (r * 4 + c)
                if i_idx < N:
                    # w[i] final = w_inicial + aux
                    w[i_idx] += aux_accum[r][c]
                    res_matrix[r][c] = int(w[i_idx])
                    op_matrix[r][c] = f"SW R3 OUT (w[{i_idx}])"
        print_cgra_block(instr_cnt, res_matrix, reg_matrix, op_matrix)
        instr_cnt += 1

def main():
    parser = argparse.ArgumentParser(description="Simulador del Algoritmo GEMVER P3 al estilo Traza CGRA")
    parser.add_argument("--npz", type=str, required=True, help="Ruta al fichero de datos .npz")
    args = parser.parse_args()

    simulate_gemver_p3_cgra(args.npz)

if __name__ == "__main__":
    main()