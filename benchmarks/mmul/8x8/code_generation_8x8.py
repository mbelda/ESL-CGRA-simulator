import csv
import sys
import os

N = 8  # CGRA size (5x5)

# Create a 5x5 matrix initialized with "NOP"
def create_cycle():
    return [["NOP" for _ in range(N)] for _ in range(N)]

# Fill a specific cell
def fill_cell(cgra, row, col, instr):
    if 0 <= row < N and 0 <= col < N:
        cgra[row][col] = instr
    else:
        print("Error: position out of range")

# Fill a circular diagonal
# diag = 1..5 (1 = main diagonal, 2 = first upper circular, ..., 5 = last circular)
def fill_diag(cgra, diag, instr):
    if not (1 <= diag <= N):
        print("Error: diagonal out of range (1..5)")
        return
    
    offset = diag - 1
    for i in range(N):
        j = (i + offset) % N
        cgra[i][j] = instr

# Fill the entire cycle with the same instruction
def fill_cycle(instr):
    return [[instr for _ in range(N)] for _ in range(N)]

# Export multiple cycles to CSV
def export_csv(cycles, filename):
    """
    cycles: list of CGRA matrices (each one 5x5)
    """
    # Create directory if it does not exist
    #os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        for idx, cgra in enumerate(cycles):
            # Cycle number line
            writer.writerow([idx] + [""] * (N - 1))

            # Instructions (5 rows)
            for row in cgra:
                complete_row = [instr if instr else "NOP" for instr in row]
                writer.writerow(complete_row)

    print(f"CSV generated: {filename}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cgra.py <output_path.csv>")
        sys.exit(1)

    output_path = sys.argv[1]

    cycles = []

    
    # PC 0: Load config values
    c0 = create_cycle()
    fill_diag(c0, 1, "LWD R2, 4") # B
    fill_diag(c0, 2, "LWD R2, 4") # C
    fill_diag(c0, 3, "LWD R2, 4") # A
    fill_diag(c0, 6, "LWD R2, 4") # C (dup)
    fill_cell(c0, 0, 3, "LWD R2, 4") # Loop 1
    fill_cell(c0, 1, 0, "LWD R2, 4") # Loop 2
    fill_cell(c0, 2, 1, "LWD R2, 4") # Loop 3
    fill_cell(c0, 3, 2, "LWD R2, 4") # Loop 2 (dup)
    fill_cell(c0, 5, 4, "LWD R2, 4") # Loop 2 (dup)
    fill_cell(c0, 7, 6, "LWD R2, 4") # Loop 2 (dup)
    cycles.append(c0)
    

    # PC 1: Load config values 
    c1 = create_cycle()
    fill_diag(c1, 1, "LWD R1, 4") # -4*colsB
    fill_diag(c1, 2, "LWD R1, 4") # colsA
    fill_diag(c1, 6, "LWD R1, 4") # 32*colsB - 32*nItL2
    cycles.append(c1)
    
    # PC 2: Initialize accumulator
    c2 = fill_cycle("SADD R3, ZERO, ZERO")
    cycles.append(c2)

    # PC 3: Load A and B values
    c3 = create_cycle()
    fill_diag(c3, 1, "LWI R0, R2")  # Load B
    fill_diag(c3, 3, "LWI R0, R2")  # Load A
    fill_cell(c3, 2, 1, "SADD R1, R1, 1") # Loop 3 ++
    cycles.append(c3)
    # D1: B, D2: -, D3: A, D4: -, D5: -, D6: -, D7: -, D8: -

    # PC 4: Share A and B values
    c4 = create_cycle()
    fill_diag(c4, 2, "SADD R0, ZERO, RCB")  # Share B
    fill_diag(c4, 8, "SADD R0, ZERO, RCT")  # Share B
    fill_diag(c4, 4, "SADD R0, ZERO, RCL")  # Share A
    cycles.append(c4)
    # D1: B, D2: B, D3: A, D4: A, D5: -, D6: -, D7: -, D8: B

    # PC 5: Share A and B values
    c5 = create_cycle()
    fill_diag(c5, 3, "SADD ROUT, ZERO, RCB")  # Share B
    fill_diag(c5, 7, "SADD R0, ZERO, RCT")  # Share B
    fill_diag(c5, 2, "SADD ROUT, ZERO, RCR")  # Share A
    fill_diag(c5, 5, "SADD R0, ZERO, RCL")  # Share A
    cycles.append(c5)
    # D1: B, D2: BA, D3: AB, D4: A, D5: A, D6: -, D7: B, D8: B

    # PC 6: Share A and B values
    c6 = create_cycle()
    fill_diag(c6, 4, "SADD ROUT, ZERO, RCB")  # Share B
    fill_diag(c6, 6, "SADD R0, ZERO, RCT")  # Share B
    fill_diag(c6, 1, "SADD ROUT, ZERO, RCR")  # Share A
    cycles.append(c6)
    # D1: BA, D2: BA, D3: AB, D4: AB, D5: A, D6: B, D7: B, D8: B

    # PC 7: Share A and B values
    c7 = create_cycle()
    fill_diag(c7, 5, "SADD ROUT, ZERO, RCB")  # Share B
    fill_diag(c7, 8, "SADD ROUT, ZERO, RCR")  # Share A
    fill_diag(c7, 6, "SADD ROUT, ZERO, RCL")  # Share A
    cycles.append(c7)
    # D1: BA, D2: BA, D3: AB, D4: AB, D5: AB, D6: BA, D7: B, D8: BA

    # PC 8: Share A and B values
    c8 = create_cycle()
    fill_diag(c8, 7, "SADD ROUT, ZERO, RCL")  # Share A
    cycles.append(c8)
    # D1: BA, D2: BA, D3: AB, D4: AB, D5: AB, D6: BA, D7: BA, D8: BA

    # PC 9: Multiply and accumulate
    c9 = fill_cycle("SMUL ROUT, R0, ROUT")
    cycles.append(c9)

    # PC 10: Accumulate result
    c10 = fill_cycle("SADD R3, R3, ROUT")
    cycles.append(c10)

    # PC 11: Update A and B address
    c11 = create_cycle()
    fill_diag(c11, 3, "SADD R2, R2, 4") # A += 4
    fill_diag(c11, 1, "SSUB R2, R2, R1") # B += 4*colsB
    fill_cell(c11, 2, 1, "BNE R1, R2, 3") # Loop 3
    cycles.append(c11)

    # PC 12: Prepare store address
    c12 = create_cycle()
    fill_diag(c12, 2, "SADD ROUT, ZERO, R2") # Show &C
    fill_diag(c12, 6, "SADD ROUT, ZERO, R2") # Show &C
    fill_cell(c12, 1, 0, "SADD R1, R1, 1") # Loop 2 ++
    cycles.append(c12)

    # PC 13: Share store address
    c13 = create_cycle()
    fill_diag(c13, 1,    "SADD ROUT, RCR, -4")
    fill_cell(c13, 7, 7, "SADD ROUT, RCR, 28")  
    fill_diag(c13, 3,    "SADD ROUT, RCL, 4")
    fill_cell(c13, 6, 0, "SADD ROUT, RCL, -28")
    fill_diag(c13, 2,    "SADD R2, R2, 32") # C += 32 for next it
    fill_diag(c13, 5,    "SADD ROUT, RCR, -4")
    fill_cell(c13, 3, 7, "SADD ROUT, RCR, 28")  
    fill_diag(c13, 7,    "SADD ROUT, RCL, 4")
    fill_cell(c13, 2, 0, "SADD ROUT, RCL, -28")
    fill_diag(c13, 6,    "SADD R2, R2, 32") # C += 32 for next it
    cycles.append(c13)

    # PC 14: Share store address
    c14 = create_cycle()
    fill_diag(c14, 4,    "SADD ROUT, RCL, 4")
    fill_cell(c14, 5, 0, "SADD ROUT, RCL, -28")
    fill_diag(c14, 8,    "SADD ROUT, RCR, -4")
    fill_cell(c14, 0, 7, "SADD ROUT, RCR, 28")
    fill_diag(c14, 2,    "SADD ROUT, R2, -32") # Actual value for &C
    fill_diag(c14, 6,    "SADD ROUT, R2, -32") # Actual value for &C
    cycles.append(c14)

    # PC 15: Store result
    c15 = fill_cycle("SWI R3, ROUT")
    cycles.append(c15)

    # PC 16: Update A and B address
    c16 = create_cycle()
    fill_diag(c16, 1, "SADD R2, R2, 32")     # B += 32 (N*sizeof(int))
    fill_diag(c16, 2, "SADD ROUT, ZERO, R1") # colsA
    cycles.append(c16)

    # PC 17: Update A and B address
    c17 = create_cycle()
    fill_diag(c17, 1, "SMUL ROUT, RCR, R1")  # -4*colsA*colsB
    fill_diag(c17, 3, "SMUL ROUT, RCL, -4")  # -4*colsA 
    fill_cell(c17, 2, 1, "SADD R1, ZERO, ZERO") # Reset Loop 3
    cycles.append(c17)

    # PC 18: Update A and B address
    c18 = create_cycle()
    fill_diag(c18, 1, "SADD R2, R2, ROUT") # B += -4*colsA*colsB
    fill_diag(c18, 3, "SADD R2, R2, ROUT") # A += -4*colsA
    fill_cell(c18, 1, 0, "SADD R3, ZERO, ZERO") # Clear accumulator
    cycles.append(c18)

    # PC 19: Clear accumulators
    c19 = fill_cycle("SADD R3, ZERO, ZERO")
    fill_cell(c19, 1, 0, "BNE R1, R2, 3") # Loop 2
    cycles.append(c19)

    # PC 20: Update A, B and C address
    c20 = create_cycle()
    fill_diag(c20, 2, "SMUL ROUT, R1, 32") # 32*colsA
    fill_cell(c20, 1, 0, "SMUL ROUT, R2, -32") # -32*nItL2
    fill_cell(c20, 3, 2, "SMUL ROUT, R2, -32") # -32*nItL2 (dup)
    fill_cell(c20, 5, 4, "SMUL ROUT, R2, -32") # -32*nItL2 (dup)
    fill_cell(c20, 7, 6, "SMUL ROUT, R2, -32") # -32*nItL2 (dup)
    fill_cell(c20, 0, 3, "SADD R1, R1, 1") # Loop 1 ++
    fill_diag(c20, 1, "SMUL ROUT, R1, -8") # -8*-4*colsB
    cycles.append(c20)

    # PC 21: Update A, B and C address
    c21 = create_cycle()
    fill_diag(c21, 3,    "SADD R2, R2, RCL")     # A += 32*colsA
    fill_diag(c21, 1,    "SADD ROUT, ZERO, RCB") # -32*nItL2
    fill_cell(c21, 1, 1, "SADD ROUT, ZERO, RCL") # -32*nItL2
    fill_cell(c21, 3, 3, "SADD ROUT, ZERO, RCL") # -32*nItL2
    fill_cell(c21, 5, 5, "SADD ROUT, ZERO, RCL") # -32*nItL2
    fill_cell(c21, 7, 7, "SADD ROUT, ZERO, RCL") # -32*nItL2
    fill_diag(c21, 2,    "SADD R2, R2, RCL")     # C += 32*colsB
    fill_cell(c21, 1, 0, "SADD R1, ZERO, ZERO")  # Reset Loop 2
    cycles.append(c21)

    # PC 22: Update A, B and C address
    c22 = create_cycle()
    fill_diag(c22, 1, "SADD R2, R2, ROUT") # B += -32*nItL2
    fill_diag(c22, 2, "SADD R2, R2, RCL")  # C += -32*nItL2
    fill_cell(c22, 0, 3, "BNE R1, R2, 3")  # Loop 1
    fill_diag(c22, 6, "SADD R2, R2, R1")   # C += 32colsB - 32nItL2
    cycles.append(c22)

    # PC 23: End program
    c23 = create_cycle()
    fill_cell(c23, 0, 0, "EXIT")
    cycles.append(c23)


    # Export with the path given as argument
    export_csv(cycles, output_path)
