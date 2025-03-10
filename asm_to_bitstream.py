import sys
import csv
import os

N_ROWS_CGRA = 4
N_COLS_CGRA = 4
N_MAX_INSTR = 512

def asm_to_hex_instruction(instruction):
    opcode_map = {
        "NOP": 0, "SADD": 1, "SSUB": 2, "SMUL": 3, "FXPMUL": 4, "SLT": 5, "SRT": 6, "SRA": 7,
        "LAND": 8, "LOR": 9, "LXOR": 10, "LNAND": 11, "LNOR": 12, "LXNOR": 13, "BSFA": 14, "BZFA": 15,
        "BEQ": 16, "BNE": 17, "BLT": 18, "BGE": 19, "JUMP": 20, "LWD": 21, "SWD": 22, "LWI": 23, "SWI": 24,
        "EXIT": 25
    }
    
    rs_map = {
        "ZERO": 0, "ROUT": 1, "RCL": 2, "RCR": 3, "RCT": 4, "RCB": 5, "R0": 6, "R1": 7, "R2": 8, "R3": 9,
        "IMM": 10
    }
    
    muxf_map = {"PREV": 0, "RCL": 1, "RCR": 2, "RCT": 3, "RCB": 4}

    rd_map = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "ROUT": 4}

    operands_type_1 = 4
    operands_type_2 = 5
    operands_type_3 = 4
    operands_type_4 = 3
    operands_type_5 = 2
    operands_type_6 = 3

    parts = instruction.replace(",", "").split()
    if not parts:
        return None
    
    alu_op = opcode_map.get(parts[0], -1)
    immediate = 0
    muxA, muxB, muxf_sel, rf_sel, rf_we = 0, 0, 0, 0, 0


    if (alu_op == 0) or (alu_op == 25): # Type 0: opcode
        pass
    elif (alu_op >= 1) & (alu_op <= 13): # TYPE 1: opcode rd, rs1, rs2
        if len(parts) < operands_type_1:
            raise ValueError(f"Not enought operands for {parts[0]}. Expected: opcode rd, rs1, rs2")
        # Dest
        rf_sel = rd_map.get(parts[1], -1)
        rf_we = 1
        if rf_sel == -1:
            raise ValueError(f"Destination register ( {parts[1]} ) not correct for {parts[0]}")
        if rf_sel == 4: # ROUT
            rf_sel = 0
            rf_we = 0
        # Source
        muxA = rs_map.get(parts[2], -1)
        if muxA == -1:
            if parts[2].lstrip('-').isdigit(): # It can be inmediate
                immediate = int(parts[2])
                muxA = rs_map.get("IMM", 0)
            else:
                raise ValueError(f"Source register ( {parts[2]} ) not correct for {parts[0]}")
        muxB = rs_map.get(parts[3], -1)
        if muxB == -1:
            if parts[3].lstrip('-').isdigit(): # It can be inmediate
                immediate = int(parts[3])
                muxB = rs_map.get("IMM", 0)
            else:
                raise ValueError(f"Source register ( {parts[3]} ) not correct for {parts[0]}")
        if (muxA == rs_map.get("IMM", 0)) & (muxB == rs_map.get("IMM", 0)):
            raise ValueError(f"Both operands cannot be immediates for {parts[0]}")
    elif (alu_op >= 14) & (alu_op <= 15): # TYPE 2: opcode rd, rs1, rs2, fs
        if len(parts) < operands_type_2:
            raise ValueError(f"Incorrect operands for {parts[0]}. Expected: opcode rd, rs1, rs2, fs")
        # Dest
        rf_sel = rd_map.get(parts[1], -1)
        rf_we = 1
        if rf_sel == -1:
            raise ValueError(f"Destination register ( {parts[1]} ) not correct for {parts[0]}")
        if rf_sel == 4: # ROUT
            rf_sel = 0
            rf_we = 0
        # Source
        muxA = rs_map.get(parts[2], -1)
        if muxA == -1:
            if parts[2].lstrip('-').isdigit(): # It can be inmediate
                immediate = int(parts[2])
                muxA = rs_map.get("IMM", 0)
            else:
                raise ValueError(f"Source register ( {parts[2]} ) not correct for {parts[0]}")
        muxB = rs_map.get(parts[3], -1)
        if muxB == -1:
            if parts[3].lstrip('-').isdigit(): # It can be inmediate
                immediate = int(parts[3])
                muxB = rs_map.get("IMM", 0)
            else:
                raise ValueError(f"Source register ( {parts[3]} ) not correct for {parts[0]}")
        if (muxA == rs_map.get("IMM", 0)) & (muxB == rs_map.get("IMM", 0)):
            raise ValueError(f"Both operands cannot be inmediates for {parts[0]}")
        # Flag
        muxf_sel = muxf_map.get(parts[4], -1)
        if muxf_sel == -1:
            raise ValueError(f"Flag option ( {parts[4]} ) not correct for {parts[0]}")
    elif (alu_op >= 16) & (alu_op <= 19): # TYPE 3: opcode rs1, rs2, addr
        if len(parts) < operands_type_3:
            raise ValueError(f"Incorrect operands for {parts[0]}. Expected: opcode rs1, rs2, addr")
        # Source
        muxA = rs_map.get(parts[1], -1)
        if muxA == -1:
            raise ValueError(f"Source register ( {parts[1]} ) not correct for {parts[0]}")
        muxB = rs_map.get(parts[2], -1)
        if muxB == -1:
            raise ValueError(f"Source register ( {parts[2]} ) not correct for {parts[0]}")
        immediate = int(parts[3])
    elif alu_op == 20: # TYPE 4: opcode rs1, rs2
        if len(parts) < operands_type_4:
            raise ValueError(f"Incorrect operands for {parts[0]}. Expected: opcode rs1, rs2")
        # Source
        muxA = rs_map.get(parts[1], -1)
        if muxA == -1:
            if parts[1].lstrip('-').isdigit(): # It can be inmediate
                immediate = int(parts[1])
                muxA = rs_map.get("IMM", 0)
            else:
                raise ValueError(f"Source register ( {parts[1]} ) not correct for {parts[0]}")
        muxB = rs_map.get(parts[2], -1)
        if muxB == -1:
            if parts[2].lstrip('-').isdigit(): # It can be inmediate
                immediate = int(parts[2])
                muxB = rs_map.get("IMM", 0)
            else:
                raise ValueError(f"Source register ( {parts[2]} ) not correct for {parts[0]}")
        if (muxA == rs_map.get("IMM", 0)) & (muxB == rs_map.get("IMM", 0)):
            raise ValueError(f"Both operands cannot be inmediates for {parts[0]}")
    elif (alu_op >= 21) & (alu_op <= 22): # TYPE 5: opcode rd
        if len(parts) < operands_type_5:
            raise ValueError(f"Incorrect operands for {parts[0]}. Expected: opcode rd")
        # Dest
        rf_sel = rd_map.get(parts[1], -1)
        rf_we = 1
        if rf_sel == -1:
            raise ValueError(f"Destination register ( {parts[1]} ) not correct for {parts[0]}")
        if rf_sel == 4: # ROUT
            rf_sel = 0
            rf_we = 0
    elif (alu_op >= 23) & (alu_op <= 24): # TYPE 5: opcode rd, rs
        if len(parts) < operands_type_6:
            raise ValueError(f"Incorrect operands for {parts[0]}. Expected: opcode rd")
        # Dest
        rf_sel = rd_map.get(parts[1], -1)
        rf_we = 1
        if rf_sel == -1:
            raise ValueError(f"Destination register ( {parts[1]} ) not correct for {parts[0]}")
        if rf_sel == 4: # ROUT
            rf_sel = 0
            rf_we = 0
        # Source
        muxA = rs_map.get(parts[2], -1)
        if muxA == -1:
            if parts[2].lstrip('-').lstrip('-').isdigit(): # It can be inmediate
                immediate = int(parts[2])
                muxA = rs_map.get("IMM", 0)
            else:
                raise ValueError(f"Source register ( {parts[2]} ) not correct for {parts[0]}")
    else:
        raise ValueError(f"Unknown opcode: {parts[0]}")

    instruction_hex = int("0", 32)

    instruction_hex = (
        (immediate & 0x1FFF) |
        ((muxf_sel & 0x7) << 13) |
        ((rf_we & 0x1) << 16) |
        ((rf_sel & 0x3) << 17) |
        ((alu_op & 0x1F) << 19) |
        ((muxB & 0xF) << 24) |
        ((muxA & 0xF) << 28)
    )
    
    return f"{instruction_hex:08X}"

def parse_kernel_config(config_file):
    config = {}
    with open(config_file, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            config[key.strip()] = int(value.strip())
    return config

def nCols_to_one_hot(x):
    if x < 1 or x > 4:
        raise ValueError("nCols debe estar entre 1 y 4")
    return (1 << x) - 1  # Desplazar 1 a la posición de x y restar 1 para que se pongan los bits a la derecha como 1


def main():
    if len(sys.argv) < 4:
        print("Uso: python asm_to_bitstream.py <input_file.csv> <config_file> <output_file>")
        return
    
    input_file = sys.argv[1]
    config_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Extraer el nombre de la última carpeta en la ruta del fichero de entrada
    kernel_name = os.path.basename(os.path.dirname(os.path.abspath(input_file))).upper()
    
    # Procesar configuración del kernel
    config = parse_kernel_config(config_file)
    nCols = config.get("nCols", 0)  # Suponiendo que el archivo de configuración tiene "nCols"
    cols_one_hot = nCols_to_one_hot(nCols)  # Convertir a codificación one-hot
    start_addr = config.get("start_addr", 0)
    n_instructions = config.get("nInstructions", 0)
    n_kernel = config.get("nKernel", 0)
    
    # Construcción de la palabra de configuración (16 bits)
    config_word = ((cols_one_hot & 0xF) << 12) | ((start_addr & 0x7F) << 5) | (n_instructions & 0x1F)

    # Lectura del archivo de instrucciones csv
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        hex_instructions = []

        for row in reader:
            # Saltar las líneas que contienen números (bloques de instrucciones)
            if row and row[0].isdigit():
                continue

            for instruction in row:  # Procesar las 4 instrucciones por línea
                if not instruction.strip() or instruction.strip().startswith("#"):  # Ignorar vacíos y comentarios
                    continue
                try:
                    hex_instructions.append(asm_to_hex_instruction(instruction.strip()))
                except ValueError as e:
                    print(f"Error en instrucción '{instruction}': {e}")

    # Reorder for easy writting
    hex_for_bitstream = []
    for c in range(N_COLS_CGRA):
        for r in range(N_ROWS_CGRA):
            for n in range(n_instructions+1):
                idx = n*16 + r*N_COLS_CGRA + c
                hex_for_bitstream.append(hex_instructions[idx])

    for _ in range(N_MAX_INSTR):
        hex_for_bitstream.append("0")


    with open(output_file, "w") as out_f:
        
        out_f.write("#ifndef _CGRA_BITSTREAM_H_\n#define _CGRA_BITSTREAM_H_\n\n")
        out_f.write("#include <stdint.h>\n\n")
        out_f.write("#include \"cgra.h\"\n\n")
        out_f.write(f"// Kernel 0 => NULL\n#define {kernel_name} {n_kernel}\n\n")

        # Escribir configuración en cgra_kmem_bitstream
        out_f.write("static uint32_t cgra_kmem_bitstream[CGRA_KMEM_SIZE] = { ")
        out_f.write("    0x0, " * n_kernel)
        out_f.write(f"    0x{config_word:04X}, ")
        out_f.write("    0x0, " * (16 - n_kernel - 1))
        out_f.write("};\n\n")

        # Escribir configuración en cgra_imem_bitstream
        out_f.write("const uint32_t cgra_imem_bitstream[CGRA_KMEM_SIZE] = { ")
        for hex_instruction in hex_for_bitstream:
            out_f.write(f"    0x{hex_instruction}, ")
        out_f.write("};\n\n")
        
        
        out_f.write("#endif // _CGRA_BITSTREAM_H_\n")
    
    print(f"Bitstream guardado en {output_file}")

if __name__ == "__main__":
    main()
