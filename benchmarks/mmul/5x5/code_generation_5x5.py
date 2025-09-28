import csv
import sys
import os

N = 5  # CGRA size (5x5)

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
    c0 = fill_cycle("LWD R2, 4") 
    fill_diag(c0, 4, "NOP")
    fill_cell(c0, 0, 3, "LWD R2, 4")
    fill_cell(c0, 4, 3, "NOP")
    cycles.append(c0)

    # PC 1: Load config values
    c1 = create_cycle()
    fill_diag(c1, 1, "LWD R1, 4")
    fill_diag(c1, 2, "LWD R1, 4")
    fill_cell(c1, 0, 3, "SADD R1, ZERO, ZERO") # Loop 1
    fill_cell(c1, 1, 0, "SADD R1, ZERO, ZERO") # Loop 2
    fill_cell(c1, 2, 1, "SADD R1, ZERO, ZERO") # Loop 3
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

    # PC 4: Share A and B values
    c4 = create_cycle()
    fill_diag(c4, 2, "SADD R0, ZERO, RCB")  # Share B
    fill_diag(c4, 5, "SADD R0, ZERO, RCT")  # Share B
    fill_diag(c4, 4, "SADD R0, ZERO, RCL")  # Share A
    cycles.append(c4)

    # PC 5: Share A and B values
    c5 = create_cycle()
    fill_diag(c5, 2, "SADD ROUT, ZERO, RCR")  # Share A
    fill_diag(c5, 3, "SADD ROUT, ZERO, RCB")  # Share B
    fill_diag(c5, 4, "SADD ROUT, ZERO, RCT")  # Share B
    fill_diag(c5, 5, "SADD ROUT, ZERO, RCL")  # Share A
    cycles.append(c5)

    # PC 6: Share A and B values
    c6 = create_cycle()
    fill_diag(c6, 1, "SADD ROUT, ZERO, RCR")  # Share A
    cycles.append(c6)

    # PC 7: Multiply and accumulate
    c7 = fill_cycle("SMUL ROUT, R0, ROUT")
    cycles.append(c7)

    # PC 8: Accumulate result
    c8 = fill_cycle("SADD R3, R3, ROUT")
    cycles.append(c8)

    # PC 9: Update A and B address
    c9 = create_cycle()
    fill_diag(c9, 3, "SADD R2, R2, 4") # A += 4
    fill_diag(c9, 1, "SSUB R2, R2, R1") # B += 4*colsB
    fill_cell(c9, 2, 1, "BNE R1, R2, 3") # Loop 3
    cycles.append(c9)

    # PC 10: Prepare store address
    c10 = create_cycle()
    fill_diag(c10, 2, "SADD ROUT, ZERO, R2") # Show &C
    fill_cell(c10, 1, 0, "SADD R1, R1, 1") # Loop 2 ++
    cycles.append(c10)

    # PC 11: Share store address
    c11 = create_cycle()
    fill_diag(c11, 1,    "SADD ROUT, RCR, -4")
    fill_cell(c11, 4, 4, "SADD ROUT, RCR, 16")  
    fill_diag(c11, 3,    "SADD ROUT, RCL, 4")
    fill_cell(c11, 3, 0, "SADD ROUT, RCL, -16")
    fill_diag(c11, 2,    "SADD R2, R2, 20") # C += 20 for next it
    cycles.append(c11)

    # PC 12: Share store address
    c12 = create_cycle()
    fill_diag(c12, 4,    "SADD ROUT, RCL, 4")
    fill_cell(c12, 2, 0, "SADD ROUT, RCL, -16")
    fill_diag(c12, 5,    "SADD ROUT, RCR, -4")
    fill_cell(c12, 0, 4, "SADD ROUT, RCR, 16")
    fill_diag(c12, 2,    "SADD ROUT, R2, -20") # Actual value for &C
    cycles.append(c12)

    # PC 13: Store result
    c13 = fill_cycle("SWI R3, ROUT")
    cycles.append(c13)

    # PC 14: Update A and B address
    c14 = create_cycle()
    fill_diag(c14, 1, "SADD R2, R2, 20")     # B += 20 (N*sizeof(int))
    fill_diag(c14, 2, "SADD ROUT, ZERO, R1") # colsA
    cycles.append(c14)

    # PC 15: Update A and B address
    c15 = create_cycle()
    fill_diag(c15, 1, "SMUL ROUT, RCR, R1")  # -4*colsA*colsB
    fill_diag(c15, 3, "SMUL ROUT, RCL, -4")  # -4*colsA 
    fill_cell(c15, 2, 1, "SADD R1, ZERO, ZERO") # Reset Loop 3
    cycles.append(c15)

    # PC 16: Update A and B address
    c16 = create_cycle()
    fill_diag(c16, 1, "SADD R2, R2, ROUT") # B += -4*colsA*colsB
    fill_diag(c16, 3, "SADD R2, R2, ROUT") # A += -4*colsA
    fill_cell(c16, 1, 0, "SADD R3, ZERO, ZERO") # Clear accumulator
    cycles.append(c16)

    # PC 17: Clear accumulators
    c17 = fill_cycle("SADD R3, ZERO, ZERO")
    fill_cell(c17, 1, 0, "BNE R1, R2, 3") # Loop 2
    cycles.append(c17)

    # PC 18: Update A and B address
    c18 = create_cycle()
    fill_diag(c18, 2, "SMUL ROUT, R1, 20") # 20*colsA
    fill_cell(c18, 1, 0, "SMUL ROUT, R2, -20") # -20*nItL2
    fill_cell(c18, 3, 2, "SMUL ROUT, R2, -20") # -20*nItL2 (dup)
    fill_cell(c18, 0, 4, "SMUL ROUT, R2, -20") # -20*nItL2 (dup)
    fill_cell(c18, 0, 3, "SADD R1, R1, 1") # Loop 1 ++
    fill_diag(c18, 1, "SMUL ROUT, R1, -5") # -5*-4*colsB
    cycles.append(c18)

    # PC 19: Update A and B address
    c19 = create_cycle()
    fill_diag(c19, 3,    "SADD R2, R2, RCL")     # A += 20*colsA
    fill_diag(c19, 1,    "SADD ROUT, ZERO, RCL") # -20*nItL2
    fill_cell(c19, 2, 2, "SADD ROUT, ZERO, RCB") # -20*nItL2
    fill_cell(c19, 4, 4, "SADD ROUT, ZERO, RCB") # -20*nItL2
    fill_diag(c19, 2,    "SADD R2, R2, RCL")     # C += 20*colsB
    fill_cell(c19, 1, 0, "SADD R1, ZERO, ZERO")  # Reset Loop 2
    cycles.append(c19)

    # PC 20: Update A and B address
    c20 = create_cycle()
    fill_diag(c20, 1, "SADD R2, R2, ROUT") # B += -20*nItL2
    fill_diag(c20, 2, "SADD R2, R2, RCL")  # C += -20*nItL2
    fill_cell(c20, 0, 3, "BNE R1, R2, 3") # Loop 1
    cycles.append(c20)

    # PC 21: End program
    c21 = create_cycle()
    fill_cell(c21, 0, 0, "EXIT")
    cycles.append(c21)

    # Export with the path given as argument
    export_csv(cycles, output_path)
