import csv
import sys
import os

N_ROWS_CGRA = 4
N_COLS_CGRA = 4

def sign_extend(value, bit_size=13):
    if value & (1 << (bit_size - 1)):
        value -= (1 << bit_size)
    return value

def hex_to_asm_instruction(hex_instruction):
    opcode_map = {
        0: "NOP", 1: "SADD", 2: "SSUB", 3: "SMUL", 4: "FXPMUL", 5: "SLT", 6: "SRT", 7: "SRA",
        8: "LAND", 9: "LOR", 10: "LXOR", 11: "LNAND", 12: "LNOR", 13: "LXNOR", 14: "BSFA", 15: "BZFA",
        16: "BEQ", 17: "BNE", 18: "BLT", 19: "BGE", 20: "JUMP", 21: "LWD", 22: "SWD", 23: "LWI", 24: "SWI",
        25: "EXIT"
    }
    
    rs_map = {
        0: "ZERO", 1: "ROUT", 2: "RCL", 3: "RCR", 4: "RCT", 5: "RCB", 6: "R0", 7: "R1", 8: "R2", 9: "R3",
        10: "IMM"
    }
    
    rd_map = {0: "R0", 1: "R1", 2: "R2", 3: "R3", 4: "ROUT"}
    muxf_map = {0: "PREV", 1: "RCL", 2: "RCR", 3: "RCT", 4: "RCB"}
    
    hex_clean = hex_instruction.strip().lower().replace("0x", "").replace(";", "")
    if not hex_clean or int(hex_clean, 16) == 0:
        return "NOP"
        
    instruction = int(hex_clean, 16)
    
    immediate = instruction & 0x1FFF
    muxf_sel = (instruction >> 13) & 0x7
    rf_we = (instruction >> 16) & 0x1
    rf_sel = (instruction >> 17) & 0x3
    alu_op = (instruction >> 19) & 0x1F
    muxB = (instruction >> 24) & 0xF
    muxA = (instruction >> 28) & 0xF

    immediate = sign_extend(immediate)
    
    opcode = opcode_map.get(alu_op, "UNKNOWN")
    rd = rd_map.get(rf_sel, "UNKNOWN")
    rs1 = rs_map.get(muxA, "UNKNOWN")
    rs2 = rs_map.get(muxB, "UNKNOWN")
    flag_config = muxf_map.get(muxf_sel, "UNKNOWN")
    
    if rf_we == 0:
        rd = "ROUT"

    if opcode in ["NOP", "EXIT"]:
        return f"{opcode}"
    elif opcode in ["SADD", "SSUB", "SMUL", "FXPMUL", "SLT", "SRT", "SRA", "LAND", "LOR", "LXOR", "LNAND", "LNOR", "LXNOR"]:
        if rs1 == "IMM": rs1 = str(immediate)
        if rs2 == "IMM": rs2 = str(immediate)
        return f"{opcode} {rd}, {rs1}, {rs2}"
    elif opcode in ["BSFA", "BZFA"]:
        return f"{opcode} {rd}, {rs1}, {rs2}, {flag_config}"
    elif opcode in ["BEQ", "BNE", "BLT", "BGE"]:
        return f"{opcode} {rs1}, {rs2}, {immediate}"
    elif opcode in ["JUMP"]:
        if rs1 == "IMM": rs1 = str(immediate)
        if rs2 == "IMM": rs2 = str(immediate)
        return f"{opcode} {rs1}, {rs2}"
    elif opcode in ["LWD", "SWD"]:
        return f"{opcode} {rd}"
    elif opcode in ["LWI", "SWI"]:
        if rs1 == "IMM": rs1 = str(immediate)
        return f"{opcode} {rd}, {rs1}"
    else:
        return f"UNKNOWN_{hex(instruction)}"

def process_csv_matrix_to_grid(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: El archivo de entrada '{input_file}' no existe.")
        sys.exit(1)

    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader) # Read columns header (Index, PE00, PE10, PE20, PE30)
        rows_data = list(reader)

    output_rows = []

    for row in rows_data:
        if not row:
            continue
        
        index_val = row[0]
        
        # 1. Write the step header row (e.g., "0,,,")
        output_rows.append([index_val, "", "", ""])
        
        # 2. Decode the execution values for the PEs at this cycle step
        # Row elements map: row[1]=PE00, row[2]=PE10, row[3]=PE20, row[4]=PE30
        pe00_asm = hex_to_asm_instruction(row[1])
        pe10_asm = hex_to_asm_instruction(row[2])
        pe20_asm = hex_to_asm_instruction(row[3])
        pe30_asm = hex_to_asm_instruction(row[4])
        
        # 3. Form the 4x4 CGRA physical grid block for this cycle
        # Column 0 gets the decoded values, Columns 1-3 get "NOP"
        output_rows.append([pe00_asm, "NOP", "NOP", "NOP"])  # CGRA Row 0
        output_rows.append([pe10_asm, "NOP", "NOP", "NOP"])  # CGRA Row 1
        output_rows.append([pe20_asm, "NOP", "NOP", "NOP"])  # CGRA Row 2
        output_rows.append([pe30_asm, "NOP", "NOP", "NOP"])  # CGRA Row 3

    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(output_rows)

    print(f"Conversión exitosa. Archivo de cuadricula CGRA guardado en: '{output_file}'")

def main():
    if len(sys.argv) != 3:
        print("Uso: python script.py <archivo_entrada.csv> <archivo_salida.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_csv_matrix_to_grid(input_file, output_file)

if __name__ == "__main__":
    main()