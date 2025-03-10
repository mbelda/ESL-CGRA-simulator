import csv
import sys

N_ROWS_CGRA = 4
N_COLS_CGRA = 4
N_MAX_INSTR = 512

# Función para manejar los números en complemento a dos, si es negativo
def sign_extend(value, bit_size=13):
    """
    Extiende el valor si es negativo (complemento a dos).
    """
    # Si el bit más significativo es 1 (número negativo en complemento a dos)
    if value & (1 << (bit_size - 1)):
        # Se extiende el signo al número completo
        value -= (1 << bit_size)
    return value

def hex_to_asm_instruction(hex_instruction):
    # Mapeo de opcodes y registros
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
    
    rd_map = {
        0: "R0", 1: "R1", 2: "R2", 3: "R3", 4: "ROUT"
    }

    muxf_map = {"PREV": 0, "RCL": 1, "RCR": 2, "RCT": 3, "RCB": 4}
    
    # Desempaquetar el valor hexadecimal
    instruction = int(hex_instruction, 16)
    
    immediate = instruction & 0x1FFF
    muxf_sel = (instruction >> 13) & 0x7
    rf_we = (instruction >> 16) & 0x1
    rf_sel = (instruction >> 17) & 0x3
    alu_op = (instruction >> 19) & 0x1F
    muxB = (instruction >> 24) & 0xF
    muxA = (instruction >> 28) & 0xF

    # Aplicar extensión de signo al inmediato, si es necesario
    immediate = sign_extend(immediate)
    
    # Identificar el opcode y los registros
    opcode = opcode_map.get(alu_op, "UNKNOWN")
    rd = rd_map.get(rf_sel, "UNKNOWN")
    rs1 = rs_map.get(muxA, "UNKNOWN")
    rs2 = rs_map.get(muxB, "UNKNOWN")
    flag_config = muxf_map.get(muxf_sel, "UNKNOWN")
    
    # Si rf_we es 0, usar "ROUT" en lugar de rd
    if rf_we == 0:
        rd = "ROUT"

    # Formar la instrucción ensamblador
    if opcode in ["NOP", "EXIT"]:
        return f"{opcode}"
    elif opcode in ["SADD", "SSUB", "SMUL", "FXPMUL", "SLT", "SRT", "SRA", "LAND", "LOR", "LXOR", "LNAND", "LNOR", "LXNOR"]:
        # Si uno de los registros es IMM, sustituir por el valor inmediato
        if rs1 == "IMM":
            rs1 = str(immediate)
        if rs2 == "IMM":
            rs2 = str(immediate)
        return f"{opcode} {rd}, {rs1}, {rs2}"
    elif opcode in ["BSFA", "BZFA"]:
        return f"{opcode} {rd}, {rs1}, {rs2}, {flag_config}"
    elif opcode in ["BEQ", "BNE", "BLT", "BGE"]:
        return f"{opcode} {rs1}, {rs2}, {immediate}"
    elif opcode in ["JUMP"]:
        if rs1 == "IMM":
            rs1 = str(immediate)
        if rs2 == "IMM":
            rs2 = str(immediate)
        return f"{opcode} {rs1}, {rs2}"
    elif opcode in ["LWD", "SWD"]:
        return f"{opcode} {rd}"
    elif opcode in ["LWI", "SWI"]:
        if rs1 == "IMM":
            rs1 = str(immediate)
        return f"{opcode} {rd}, {rs1}"
    else:
        return "UNKNOWN INSTRUCTION"


def getKernelInfo(hex_info):
    # Convertir el string hexadecimal a un número entero
    value = int(hex_info, 16)

    # Extraer los campos según la distribución de bits
    columns = (value >> 12) & 0xF  # Bits 15:12 (4 bits)
    kernel_start_addr = (value >> 5) & 0x7F  # Bits 11:5 (7 bits)
    num_instr = value & 0x1F  # Bits 4:0 (5 bits)

    return num_instr

# Función principal para procesar el archivo bitstream.h y escribir a CSV
def process_bitstream(input_file, output_file):
    # Abrir el archivo de entrada y leer los valores hexadecimales
    with open(input_file, 'r') as f:
        # Leer el contenido del archivo, y extraer el array `cgra_imem_bitstream`
        content = f.read()

        start_index = content.find("cgra_kmem_bitstream")  # Encontrar el inicio del array
        if start_index == -1:
            print("No se encontró el array 'cgra_kmem_bitstream' en el archivo.")
            return

        # Buscar el array entre los corchetes
        start_index = content.find("{", start_index)
        end_index = content.find("}", start_index)
        kmem_data = content[start_index + 1:end_index].strip()
        kernel_values = kmem_data.split(",")

        start_index = content.find("cgra_imem_bitstream")  # Encontrar el inicio del array
        if start_index == -1:
            print("No se encontró el array 'cgra_imem_bitstream' en el archivo.")
            return

        # Buscar el array entre los corchetes
        start_index = content.find("{", start_index)
        end_index = content.find("}", start_index)
        bitstream_data = content[start_index + 1:end_index].strip()

        # Dividir los valores hexadecimales en una lista
        hex_values = bitstream_data.split(",")

    # Convertir cada valor hexadecimal a su formato ensamblador
    nInstructions = getKernelInfo(next(filter(lambda x: x!= "0x0", kernel_values), None))
    nInstructions+=1
    # Convertir cada valor hexadecimal a su formato ensamblador
    asm_instructions = [hex_to_asm_instruction(hex_value.strip()) for hex_value in hex_values]
    
    # Escribir el archivo CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        block_number = 0

        for n in range(nInstructions):
            # Escribir bloque
            writer.writerow([n])
            for r in range(N_ROWS_CGRA):
                # Escribir cada fila de 4 instrucciones
                row = []
                for c in range(N_COLS_CGRA):
                    idx = (r*N_COLS_CGRA + c)*nInstructions + n
                    row.append(asm_instructions[idx])
                writer.writerow(row)


    print(f"Conversión completada. Las instrucciones se han guardado en '{output_file}'.")


def process_array(input_file, output_file):
    # Abrir el archivo de entrada y leer los valores hexadecimales
    with open(input_file, 'r') as f:
        # Leer el contenido del archivo
        hex_values = f.read().strip().split(",")
    
    # Convertir cada valor hexadecimal a su formato ensamblador
    asm_instructions = [hex_to_asm_instruction(hex_value.strip()) for hex_value in hex_values]

    # Escribir el archivo de salida
    with open(output_file, 'w', newline='') as f:
        for instr in asm_instructions:
            f.write(instr + "\n")
    print(f"Conversión completada. Las instrucciones se han guardado en '{output_file}'.")
    
# Función main que maneja los argumentos de la línea de comandos
def main():
    if len(sys.argv) != 4:
        print("Uso: python script.py <archivo_entrada> <archivo_salida> B/A")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if (sys.argv[3] == "B"):
        process_bitstream(input_file, output_file)
    elif (sys.argv[3] == "A"):
        process_array(input_file, output_file)

if __name__ == "__main__":
    main()