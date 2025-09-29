import csv
import sys
import os

N = 3  # CGRA size (5x5)

# Create a NxN matrix initialized with "NOP"
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
    c0 = fill_cycle("LWD R2, 4") # D1: &A, D2: &B, D3: Loops     
    cycles.append(c0)
    
    # PC 1: Load config values 
    c1 = create_cycle()
    fill_diag(c1, 1, "LWD R1, 4") # &C
    fill_cell(c1, 0, 1, "LWD R1, 4") # colsA
    fill_cell(c1, 1, 2, "LWD R1, 4") # colsB
    fill_diag(c1, 3, "SADD R1, ZERO, ZERO") # Loops
    cycles.append(c1)
    
    # PC 2: Initialize accumulator
    c2 = fill_cycle("SADD R3, ZERO, ZERO")
    cycles.append(c2)

    # PC 3: Load A and B values
    c3 = create_cycle()
    fill_diag(c3, 1, "LWI R0, R2")  # Load A
    fill_diag(c3, 2, "LWI R0, R2")  # Load B
    fill_cell(c3, 2, 1, "SADD R1, R1, 1") # Loop 3 ++
    cycles.append(c3)
    # D1: A, D2: B, D3: -

    # PC 4: Share A and B values
    c4 = create_cycle()
    fill_diag(c4, 1, "SADD ROUT, ZERO, RCT")  # Share B
    fill_diag(c4, 2, "SADD ROUT, ZERO, RCL")  # Share A
    fill_diag(c4, 3, "SADD R0, ZERO, RCB")    # Share B
    cycles.append(c4)
    # D1: AB, D2: BA, D3: B

    # PC 5: Share A and B values
    c5 = create_cycle()
    fill_diag(c5, 3, "SADD ROUT, ZERO, RCL")  # Share A
    cycles.append(c5)
    # D1: AB, D2: BA, D3: BA

    # PC 6: Multiply and accumulate
    c6 = fill_cycle("SMUL ROUT, R0, ROUT")
    cycles.append(c6)

    # PC 7: Accumulate result
    c7 = fill_cycle("SADD R3, R3, ROUT")
    cycles.append(c7)

    # PC 8: Update A and B address
    c8 = create_cycle()
    fill_diag(c8, 1, "SADD R2, R2, 4") # A += 4
    fill_cell(c8, 1, 2, "SMUL ROUT, R1, 4") # 4*colsB
    cycles.append(c8)

    # PC 9: Update A and B address
    c9 = create_cycle()
    fill_cell(c9, 0, 2, "SADD ROUT, ZERO, RCB") # 4*colsB
    fill_cell(c9, 2, 2, "SADD ROUT, ZERO, RCT") # 4*colsB
    cycles.append(c9)

    # PC 10: Update A and B address
    c10 = create_cycle()
    fill_cell(c10, 0, 1, "SADD R2, R2, RCR")  # B += 4*colsB
    fill_cell(c10, 1, 2, "SADD R2, R2, ROUT") # B += 4*colsB
    fill_cell(c10, 2, 0, "SADD R2, R2, RCL")  # B += 4*colsB
    fill_cell(c10, 2, 1, "BNE R1, R2, 3") # Loop 3
    cycles.append(c10)

    # PC 11: Prepare store address
    c11 = create_cycle()
    fill_diag(c11, 1, "SADD R1, R1, 12") # C += 12 for next it
    fill_cell(c11, 1, 0, "SADD R1, R1, 1") # Loop 2 ++
    cycles.append(c11)

    # PC 12: Share store address
    c12 = create_cycle()
    fill_diag(c12, 1, "SADD ROUT, R1, -12") # Actual value for &C
    fill_diag(c12, 2, "SADD ROUT, RCL, -8")
    fill_cell(c12, 2, 0, "SADD ROUT, RCL, -20")
    fill_diag(c12, 3, "SADD ROUT, RCR, -16")
    fill_cell(c12, 0, 2, "SADD ROUT, RCR, -4")
    cycles.append(c12)

    # PC 13: Store result
    c13 = fill_cycle("SWI R3, ROUT")
    cycles.append(c13)

    # PC 14: Update A and B address
    c14 = create_cycle()
    fill_cell(c14, 0, 1, "SMUL ROUT, R1, -4")   # -4*colsA
    fill_cell(c14, 1, 2, "SADD ROUT, R1, ZERO") # colsB
    fill_cell(c14, 2, 1, "SADD R1, ZERO, ZERO") # Reset Loop 3
    cycles.append(c14)

    # PC 15: Update A and B address
    c15 = create_cycle()
    fill_cell(c15, 0, 0, "SADD R2, R2, RCR")     # A += -4colsA
    fill_cell(c15, 1, 1, "SADD R2, R2, RCT")     # A += -4colsA
    fill_cell(c15, 2, 1, "SADD ROUT, ZERO, RCB") # -4colsA
    fill_diag(c15, 2,    "SADD R2, R2, 12")      # B += 12 (N*sizeof(int))
    fill_cell(c15, 0, 2, "SMUL ROUT, RCL, RCB")  # -4*colsA*colsB
    cycles.append(c15)

    # PC 16: Update A and B address
    c16 = create_cycle()
    fill_cell(c16, 2, 2, "SADD R2, R2, RCL")    # A += -4colsA
    fill_cell(c16, 0, 0, "SADD ROUT, ZERO, RCL") # -4colsA*colsB
    fill_cell(c16, 0, 1, "SADD R2, R2, RCR")     # B += -4*colsA*colsB
    fill_cell(c16, 1, 2, "SADD R2, R2, RCT")     # B += -4*colsA*colsB
    cycles.append(c16)

    # PC 17: Update A and B address
    c17 = create_cycle()
    fill_cell(c17, 2, 0, "SADD R2, R2, RCB")     # B += -4*colsA*colsB
    fill_cell(c17, 1, 0, "SADD R3, ZERO, ZERO")  # Clear accumulator
    cycles.append(c17)

    # PC 18: Clear accumulators
    c18 = fill_cycle("SADD R3, ZERO, ZERO")
    fill_cell(c18, 1, 0, "BNE R1, R2, 3") # Loop 2
    cycles.append(c18)

    # PC 19: Update A, B and C address
    c19 = create_cycle()
    fill_cell(c19, 0, 1, "SMUL ROUT, R1, 12")  # 12colsA
    fill_cell(c19, 1, 0, "SMUL ROUT, R2, -12") # -12nItL2
    fill_cell(c19, 1, 2, "SMUL ROUT, R1, 12")  # 12colsB
    fill_cell(c19, 0, 2, "SADD R1, R1, 1")     # Loop 1 ++
    cycles.append(c19)

    # PC 20: Update A, B and C address
    c20 = create_cycle()
    fill_cell(c20, 0, 0, "SADD R2, R2, RCR")     # A += 12colsA
    fill_cell(c20, 1, 0, "SADD ROUT, ZERO, RCL") # 12colsB
    fill_cell(c20, 1, 1, "SADD R2, R2, RCT")     # A += 12colsA
    fill_cell(c20, 1, 2, "SADD ROUT, ZERO, RCR") # -12nItL2
    fill_cell(c20, 2, 0, "SADD ROUT, ZERO, RCT") # -12nItL2
    fill_cell(c20, 2, 1, "SADD ROUT, ZERO, RCB") # 12colsA
    fill_cell(c20, 2, 2, "SADD R1, R1, RCT")     # C += 12colsB
    cycles.append(c20)

    # PC 21: Update A, B and C address
    c21 = create_cycle()
    fill_cell(c21, 0, 0, "SADD ROUT, ZERO, RCT") # -12nItL2
    fill_cell(c21, 1, 1, "SADD R1, R1, RCL")     # C += 12colsB
    fill_cell(c21, 2, 2, "SADD R2, R2, RCL")     # A += 12colsA
    cycles.append(c21)

    # PC 22: Update A, B and C address
    c22 = create_cycle()
    fill_cell(c22, 0, 0, "SADD R1, R1, ROUT") # C += -12nItL2
    fill_cell(c22, 0, 1, "SADD R2, R2, RCL")  # B += -12nItL2
    fill_cell(c22, 1, 1, "SADD R1, R1, RCR")  # C += -12nItL2
    fill_cell(c22, 1, 2, "SADD R2, R2, ROUT") # B += -12nItL2
    fill_cell(c22, 2, 0, "SADD R2, R2, ROUT") # B += -12nItL2
    fill_cell(c22, 2, 2, "SADD R1, R1, RCT")  # C += -12nItL2
    cycles.append(c22)

    # PC 23: Update A, B and C address
    c23 = create_cycle()
    fill_cell(c23, 0, 0, "SADD R1, R1, RCB") # C += 12colsB
    fill_cell(c23, 0, 2, "BNE R1, R2, 3")    # Loop 1
    fill_cell(c23, 1, 0, "SADD R1, ZERO, ZERO")  # Reset Loop 2
    cycles.append(c23)

    # PC 24: End program
    c24 = create_cycle()
    fill_cell(c24, 0, 0, "EXIT")
    cycles.append(c24)

    # Export with the path given as argument
    export_csv(cycles, output_path)
