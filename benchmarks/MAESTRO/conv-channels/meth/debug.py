#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import numpy as np
from pathlib import Path

def print_cgra_block(file_out, instr_id, matrix_data, reg_data, op_strings):
    """Escribe la salida de simulación con un formato ultra-compacto."""
    file_out.write(f"Instr =  {instr_id} ( {instr_id} )\n")
    for row in range(4):
        # Reducimos anchos: datos de matriz a 6 chars, registros a 3, ops a 18
        m_str = "[" + ",".join(f"{matrix_data[row][col]:>6}" for col in range(4)) + "]"
        r_str = "[" + ",".join(f"{reg_data[row][col]:>3}" for col in range(4)) + "]"
        o_str = "[" + ",".join(f"{op_strings[row][col]:<18}" for col in range(4)) + "]"
        # Separación mínima de 2 espacios entre bloques
        file_out.write(f"{m_str}  {r_str}  {o_str}\n")
    file_out.write("Aprox cycles this pc: 1\n")
    file_out.write("-------\n")

def simulate_3loops_cgra(npz_path, channels, kh, kw, ih, iw):
    if not os.path.exists(npz_path):
        print(f"Error: No se encuentra el fichero {npz_path}")
        return

    # Carga de datos de la convolución
    data = np.load(npz_path)
    input_lider = data["input_lider"].astype(int)
    weights = data["weights"].astype(int)
    expected_sum = int(data["expected_sum"])

    output_filename = f"debug_cgra_c{channels}_kh{kh}_kw{kw}_iw{iw}_out.log"
    f_out = open(output_filename, "w")

    # Mapeo de offsets de los arrays por canales
    tam_ch_in = ih * iw
    tam_ch_w = kh * kw

    # Cada celda PE acumulará su valor local en una matriz interna de acumuladores
    pe_accum = [[0]*4 for _ in range(4)]
    reg_matrix = [[0]*4 for _ in range(4)] 
    instr_cnt = 1

    # ==========================================================================
    # 1. BUCLE DE PROCESAMIENTO LOCAL POR CANALES (Salto de 16 en 16)
    # ==========================================================================
    f_out.write("=== EMPEZANDO BUCLE DE COMPUTACIÓN LOCAL ===\n\n")
    
    max_local_channels = (channels + 15) // 16

    for block in range(max_local_channels):
        for kh_idx in range(kh):
            for kw_idx in range(kw):
                
                # ----------------------------------------------------
                # PASO A: Lectura de Pixeles de Entrada
                # ----------------------------------------------------
                val_matrix = [[0]*4 for _ in range(4)]
                op_matrix = [["NOP"]*4 for _ in range(4)]
                
                for r in range(4):
                    for c in range(4):
                        pe_id = r * 4 + c
                        curr_ch = block * 16 + pe_id
                        
                        if curr_ch < channels:
                            pixel_idx = curr_ch * tam_ch_in + (kh_idx * iw + kw_idx)
                            if pixel_idx < len(input_lider):
                                val_matrix[r][c] = int(input_lider[pixel_idx])
                                op_matrix[r][c] = f"LWD R1(Im[{curr_ch}])"
                                
                print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
                instr_cnt += 1
                
                pixels_step = [row[:] for row in val_matrix]

                # ----------------------------------------------------
                # PASO B: Lectura de Pesos
                # ----------------------------------------------------
                val_matrix = [[0]*4 for _ in range(4)]
                op_matrix = [["NOP"]*4 for _ in range(4)]
                
                for r in range(4):
                    for c in range(4):
                        pe_id = r * 4 + c
                        curr_ch = block * 16 + pe_id
                        
                        if curr_ch < channels:
                            weight_idx = curr_ch * tam_ch_w + (kh_idx * kw + kw_idx)
                            if weight_idx < len(weights):
                                val_matrix[r][c] = int(weights[weight_idx])
                                op_matrix[r][c] = f"LWD R2(F[{curr_ch}])"
                                
                print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
                instr_cnt += 1
                
                weights_step = [row[:] for row in val_matrix]

                # ----------------------------------------------------
                # PASO C: Multiplicación
                # ----------------------------------------------------
                mult_matrix = [[0]*4 for _ in range(4)]
                op_matrix = [["NOP"]*4 for _ in range(4)]
                
                for r in range(4):
                    for c in range(4):
                        pe_id = r * 4 + c
                        curr_ch = block * 16 + pe_id
                        if curr_ch < channels:
                            mult_matrix[r][c] = pixels_step[r][c] * weights_step[r][c]
                            op_matrix[r][c] = f"SMUL R3 R1 R2"
                            
                print_cgra_block(f_out, instr_cnt, mult_matrix, reg_matrix, op_matrix)
                instr_cnt += 1

                # ----------------------------------------------------
                # PASO D: Acumulación local en cada PE
                # ----------------------------------------------------
                op_matrix = [["NOP"]*4 for _ in range(4)]
                
                for r in range(4):
                    for c in range(4):
                        pe_id = r * 4 + c
                        curr_ch = block * 16 + pe_id
                        if curr_ch < channels:
                            pe_accum[r][c] += mult_matrix[r][c]
                            op_matrix[r][c] = f"SADD R0 R0 R3"
                            
                print_cgra_block(f_out, instr_cnt, pe_accum, reg_matrix, op_matrix)
                instr_cnt += 1

    # ==========================================================================
    # 2. REDUCCIÓN / INTERCONEXIÓN ENTRE PEs
    # ==========================================================================
    f_out.write("\n=== INICIANDO REDUCCIÓN INTER-PE ===\n\n")

    val_matrix = [[0]*4 for _ in range(4)]
    op_matrix = [["NOP"]*4 for _ in range(4)]
    
    for r in range(4):
        op_matrix[r][3] = f"RCR R0 -> C2"
        val_matrix[r][3] = pe_accum[r][3]
        
        pe_accum[r][2] += pe_accum[r][3]
        op_matrix[r][2] = f"SADD R0 R0 RCR"
        val_matrix[r][2] = pe_accum[r][2]

        op_matrix[r][0] = f"RCL R0 -> C1"
        val_matrix[r][0] = pe_accum[r][0]

        pe_accum[r][1] += pe_accum[r][0]
        op_matrix[r][1] = f"SADD R0 R0 RCL"
        val_matrix[r][1] = pe_accum[r][1]

    print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
    instr_cnt += 1

    val_matrix = [[0]*4 for _ in range(4)]
    op_matrix = [["NOP"]*4 for _ in range(4)]
    for r in range(4):
        op_matrix[r][2] = f"RCR R0 -> C1"
        val_matrix[r][2] = pe_accum[r][2]
        
        pe_accum[r][1] += pe_accum[r][2]
        op_matrix[r][1] = f"SADD R0 R0 RCR"
        val_matrix[r][1] = pe_accum[r][1]
    
    print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
    instr_cnt += 1

    val_matrix = [[0]*4 for _ in range(4)]
    op_matrix = [["NOP"]*4 for _ in range(4)]
    
    op_matrix[0][1] = f"RCT R0 -> F1"
    val_matrix[0][1] = pe_accum[0][1]
    pe_accum[1][1] += pe_accum[0][1]
    
    op_matrix[2][1] = f"RCB R0 -> F1"
    val_matrix[2][1] = pe_accum[2][1]
    pe_accum[1][1] += pe_accum[2][1]

    op_matrix[3][1] = f"RCB R0 -> F2"
    val_matrix[3][1] = pe_accum[3][1]
    pe_accum[1][1] += pe_accum[3][1]

    op_matrix[1][1] = f"SADD R0 (REDUC)"
    val_matrix[1][1] = pe_accum[1][1]

    print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
    instr_cnt += 1

    # ----------------------------------------------------
    # 3. VOLCADO FINAL: SWD
    # ----------------------------------------------------
    val_matrix = [[0]*4 for _ in range(4)]
    op_matrix = [["NOP"]*4 for _ in range(4)]
    
    val_matrix[1][1] = pe_accum[1][1]
    op_matrix[1][1] = f"SWD R0 (STORE)"

    print_cgra_block(f_out, instr_cnt, val_matrix, reg_matrix, op_matrix)
    instr_cnt += 1

    f_out.close()
    
    print(f"Traza de simulación de debug guardada con éxito en: {output_filename}")
    print(f"Suma simulada obtenida en PE[1][1]: {pe_accum[1][1]} | Esperada: {expected_sum}")
    if pe_accum[1][1] == expected_sum:
        print("¡VERIFICACIÓN EXITOSA DEL MODELO DE DEBUG!")
    else:
        print("¡ERROR! El modelo matemático difiere del valor esperado de simulación.")

def main():
    parser = argparse.ArgumentParser(description="Script de simulación de Debug paso a paso para CGRA 3-loops.")
    parser.add_argument('--channels', type=int, required=True, help='Número de canales (C)')
    parser.add_argument('--kh', type=int, required=True, help='Kernel Height (KH)')
    parser.add_argument('--kw', type=int, required=True, help='Kernel Width (KW)')
    parser.add_argument('--ih', type=int, required=True, help='Input Height (IH)')
    parser.add_argument('--iw', type=int, required=True, help='Input Width (IW)')
    parser.add_argument('--npz', type=str, required=True, help='Ruta al fichero de datos .npz generado')

    args = parser.parse_args()

    simulate_3loops_cgra(args.npz, args.channels, args.kh, args.kw, args.ih, args.iw)

if __name__ == "__main__":
    main()