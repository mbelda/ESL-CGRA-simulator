import sys

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
        if rs1 == "IMM": rs1 = str(immediate)
        return f"{opcode} {rd}, {rs1}"
    else:
        return f"UNKNOWN_{hex(instruction)}"


def getKernelInfo(hex_info):
    hex_clean = hex_info.strip().lower().replace("0x", "").replace(";", "")
    value = int(hex_clean, 16)
    
    active_cols_mask = (value >> 12) & 0xF
    kernel_start_addr = (value >> 5) & 0x7F  
    num_instr = value & 0x1F  
    
    return num_instr, kernel_start_addr, active_cols_mask


def process_bitstream(hex_values):
    """
    Simulates finding kmem and imem within the passed parameter array.
    For this implementation, it assumes the entire array contains IMEM instructions, 
    and mocks kernel settings, or parses specific elements if structured.
    """
    asm_instructions = [hex_to_asm_instruction(hv) for hv in hex_values]
    
    # Example: Mocking a kernel entry using the first non-zero configuration item 
    # to maintain compatibility with your original logic block structure.
    # If your input array contains mixed KMEM data, customize this selection.
    sample_k_hex = "7A0B0001" 
    
    num_instr, start_addr, cols_mask = getKernelInfo(sample_k_hex)
    active_columns = [c for c in range(N_COLS_CGRA) if (cols_mask & (1 << c))]
    nInstructions = num_instr + 1  
    WORDS_PER_COL_WINDOW = 128

    print("\n" + "="*50)
    print(f" PROCESSING BITSTREAM TERMINAL OUTPUT ")
    print(f"Active Columns: {active_columns} | Base Address: {start_addr} | Instructions: {nInstructions}")
    print("="*50)

    for n in range(nInstructions):
        print(f"\n[Cycle/Instruction Step {n}]")
        
        for r in range(N_ROWS_CGRA):
            row = []
            for c in range(N_COLS_CGRA):
                if c not in active_columns:
                    row.append("NOP")
                else:
                    column_window_base = c * WORDS_PER_COL_WINDOW
                    idx = column_window_base + start_addr + (r * nInstructions) + n
                    
                    if idx < len(asm_instructions):
                        row.append(asm_instructions[idx])
                    else:
                        row.append("NOP")
            
            # Print row aligned as a clean table row
            print("  |  ".join(f"{inst:<15}" for inst in row))


def process_array(hex_values):
    """Converts the array of hex values to ASM and prints to the terminal."""
    asm_instructions = [hex_to_asm_instruction(hv) for hv in hex_values]
    
    print("\n" + "="*40)
    print(f"{'Index':<8} | {'Hex Value':<12} | {'ASM Instruction'}")
    print("="*40)
    for idx, (hv, instr) in enumerate(zip(hex_values, asm_instructions)):
        print(f"[{idx}]".ljust(8) + f" | {hv:<12} | {instr}")
    print("="*40 + "\n")
    

def main():
    if len(sys.argv) < 3:
        print("Uso: python script.py <Modo: B/A> <Hex1> <Hex2> ... <HexN>")
        print("Ejemplo: python script.py A 00A90000 7A0B0001 76900006")
        sys.exit(1)

    mode = sys.argv[1].upper()
    hex_arguments = sys.argv[2:]

    if mode == "B":
        process_bitstream(hex_arguments)
    elif mode == "A":
        process_array(hex_arguments)
    else:
        print("Modo inválido. Use 'A' para procesamiento de Array simple o 'B' para Bitstream.")

if __name__ == "__main__":
    main()