import csv
import sys
import os

N_ROWS_CGRA = 4
N_COLS_CGRA = 4
WORDS_PER_ROW_BANK = 128 # The bitstream splits into 128-word blocks per physical ROW

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
    if not hex_clean:
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
        # Primero procesamos si alguno es inmediato
        if rs1 == "IMM": rs1 = str(immediate)
        if rs2 == "IMM": rs2 = str(immediate)
        
        # Evaluamos cuál de los dos campos (muxA o muxB) se está usando.
        # Si muxA no es cero (es decir, no es "ZERO"), ese es nuestro input.
        if muxA != 0:
            input_reg = rs1
        # Si muxA es cero pero muxB no lo es, usamos muxB.
        elif muxB != 0:
            input_reg = rs2
        # Si ambos son cero, por defecto cae en rs1 ("ZERO") o el inmediato si correspondía
        else:
            input_reg = rs1

        return f"{opcode} {rd}, {input_reg}"
    else:
        return f"UNKNOWN_{hex(instruction)}"


def getKernelInfo(hex_info):
    hex_clean = hex_info.strip().lower().replace("0x", "").replace(";", "")
    value = int(hex_clean, 16)
    
    # Bits 15:12 -> Active Columns One-Hot mask
    active_cols_mask = (value >> 12) & 0xF
    # Bits 11:5 -> Starting address pointer index in IMEM
    kernel_start_addr = (value >> 5) & 0x7F  
    # Bits 4:0 -> Loop count configuration limit (0-indexed max index)
    num_instr = value & 0x1F  
    
    return num_instr, kernel_start_addr, active_cols_mask


def process_bitstream(input_file, output_file_base):
    with open(input_file, 'r') as f:
        content = f.read()

        # Extract KMEM data
        start_index = content.find("cgra_kmem_bitstream")
        if start_index == -1:
            print("No se encontró el array 'cgra_kmem_bitstream'.")
            return
        start_index = content.find("{", start_index)
        end_index = content.find("}", start_index)
        kmem_data = content[start_index + 1:end_index].strip()
        kernel_values = [v.strip() for v in kmem_data.split(",") if v.strip()]

        # Extract IMEM data
        start_index = content.find("cgra_imem_bitstream")
        if start_index == -1:
            print("No se encontró el array 'cgra_imem_bitstream'.")
            return
        start_index = content.find("{", start_index)
        end_index = content.find("}", start_index)
        bitstream_data = content[start_index + 1:end_index].strip()
        hex_values = [v.strip() for v in bitstream_data.split(",") if v.strip()]

    asm_instructions = [hex_to_asm_instruction(hv) for hv in hex_values]
    
    for k_idx, k_hex in enumerate(kernel_values):
        hex_clean = k_hex.strip().lower().replace("0x", "").replace(";", "")
        if not hex_clean or int(hex_clean, 16) == 0:
            continue
            
        num_instr, start_addr, cols_mask = getKernelInfo(k_hex)
        
        # Identify active columns via decoded mask (ordered left-to-right)
        active_columns = [c for c in range(N_COLS_CGRA) if (cols_mask & (1 << c))]
        nInstructions = num_instr + 1  
        
        base, ext = os.path.splitext(output_file_base)
        if not ext: ext = '.csv'
        output_file = f"{base}_kernel_{k_idx}{ext}"
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Print a quick summary header inside our generated CSV file
            writer.writerow([f"# Kernel {k_idx} Execution Layout"])
            writer.writerow([f"# Active Columns: {active_columns} | Instructions per PE: {nInstructions}"])
            writer.writerow([])

            # Step through execution cycles 0 to (nInstructions - 1)
            for n in range(nInstructions):
                writer.writerow([f"Cycle/Instruction Step {n}"])
                
                # Render Row by Row for spatial grid display mapping
                for r in range(N_ROWS_CGRA):
                    row_data = []
                    
                    for c in range(N_COLS_CGRA):
                        if c not in active_columns:
                            row_data.append("NOP")
                        else:
                            # 1. Base bank offset for row block
                            row_bank_base = r * WORDS_PER_ROW_BANK
                            
                            # 2. Sequential order position of this column among the active ones
                            active_col_index = active_columns.index(c)
                            
                            # 3. Calculate absolute array location index
                            idx = row_bank_base + start_addr + (active_col_index * nInstructions) + n
                            
                            if idx < len(asm_instructions):
                                row_data.append(asm_instructions[idx])
                            else:
                                row_data.append("NOP")
                    writer.writerow(row_data)
                writer.writerow([]) # Print gap spacing between step sequences
                    
        print(f"Kernel {k_idx} guardado en '{output_file}' (Columnas activas: {active_columns}, Dirección base: {start_addr}, Num Instr: {nInstructions}).")


def process_array(input_file, output_file):
    with open(input_file, 'r') as f:
        hex_values = [v.strip() for v in f.read().strip().split(",") if v.strip()]
    
    asm_instructions = [hex_to_asm_instruction(hv) for hv in hex_values]

    with open(output_file, 'w', newline='') as f:
        for instr in asm_instructions:
            f.write(instr + "\n")
    print(f"Conversión completada. Las instrucciones se han guardado en '{output_file}'.")
    

def main():
    if len(sys.argv) != 4:
        print("Uso: python script.py <archivo_entrada> <archivo_salida> B/A")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if sys.argv[3] == "B":
        process_bitstream(input_file, output_file)
    elif sys.argv[3] == "A":
        process_array(input_file, output_file)

if __name__ == "__main__":
    main()