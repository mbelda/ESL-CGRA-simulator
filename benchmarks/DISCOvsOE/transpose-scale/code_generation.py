import csv
import sys
import os

N = 4  # CGRA size 

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
# diag = 1..N (1 = main diagonal, 2 = first upper circular, ..., N = last circular)
def fill_diag(cgra, diag, instr):
    if not (1 <= diag <= N):
        print(f"Error: diagonal out of range (1..{N})")
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

    ## Initial configuration

    # PC 0: Load config values (&in) (RC00: nItLoopBlocksCols) (RC11: nItLoopBlocksRows)
    c0 = fill_cycle("LWD R0, 4") 
    cycles.append(c0)

    # PC 1: Load config values (&out) 
    c1 = fill_cycle("LWD R1, 4")
    cycles.append(c1)
    
    # PC 2: Load config values (shift_scale)
    c2 = fill_cycle("LWD R2, 4")
    cycles.append(c2)

    # PC 3: Load config values (colsIn, colsOut)
    c3 = create_cycle()
    fill_diag(c3, 2, "LWD R3, 4") # colsIn
    fill_diag(c3, 3, "LWD R3, 4") # colsOut
    fill_diag(c3, 4, "LWD R3, 4") # 4*(nBlocksColsi + 1)
    cycles.append(c3)

    ## Computation

    # PC 4: Prepare load address for loop control RCs (RC00 and RC11)
    c4 = create_cycle()
    fill_cell(c4, 0, 0, "SADD R3, R3, 1")
    fill_cell(c4, 0, 1, "SSUB ROUT, R0, 4")
    fill_cell(c4, 1, 2, "SSUB ROUT, R0, 4")
    cycles.append(c4)

    # PC 5: load input
    c5 = fill_cycle("LWI ROUT, R0")
    fill_cell(c5, 0, 0, "LWI ROUT, RCR")
    fill_cell(c5, 1, 1, "LWI ROUT, RCR")
    cycles.append(c5)

    # PC 6: scale
    c6 = fill_cycle("SRA ROUT, ROUT, R2")
    cycles.append(c6)

    # PC 7: store transpose
    c7 = fill_cycle("SWI ROUT, R1")
    cycles.append(c7)

    ## Inner loop address update

    ## Update out address

    # PC 8: expose 16*colsOut
    c8 = create_cycle()
    fill_diag(c8, 3, "SMUL ROUT, R3, 16")
    cycles.append(c8)

    # PC 9
    c9 = create_cycle()
    fill_diag(c9, 2, "SADD ROUT, RCR, ZERO")
    fill_diag(c9, 4, "SADD ROUT, RCL, ZERO")
    cycles.append(c9)

    # PC 10
    c10 = create_cycle()
    fill_diag(c10, 1, "SADD ROUT, RCL, ZERO")
    cycles.append(c10)

    # PC 11: out += 16*colsOut
    c11 = fill_cycle("SADD R1, R1, ROUT")
    cycles.append(c11)

    ## Update input address
    # PC 12: update input and loop branch → PC 4
    c12 = fill_cycle("SADD R0, R0, 16") # in += 16
    fill_cell(c12, 0, 0, "BNE R0, R3, 4")
    fill_cell(c12, 1, 1, "NOP")
    cycles.append(c12)


    ## Outer loop

    # Update In address
    # PC 13: expose (nBlocksColsi+1)*16 and 16*nColsi
    c13 = create_cycle()
    fill_diag(c13, 4, "SMUL ROUT, R3, 4") # Expose (nBlocksColsi + 1)*16
    fill_diag(c13, 2, "SMUL ROUT, R3, 16") # Expose 16*nColsi
    cycles.append(c13)

    # PC 14: compute offset
    c14 = create_cycle()
    fill_diag(c14, 1, "SSUB ROUT, RCR, RCL")
    fill_diag(c14, 3, "SSUB ROUT, RCL, RCR")
    cycles.append(c14)

    # PC 15: R0 += offset
    c15 = create_cycle()
    fill_diag(c15, 2, "SADD R0, R0, RCR")
    fill_diag(c15, 3, "SADD R0, R0, ROUT")
    fill_diag(c15, 4, "SADD R0, R0, RCL")
    cycles.append(c15)


    # PC 13: Update out address
    # Out += 16 - 4*nColso*(nBlocksColsi +1)
    # PC 16
    c16 = create_cycle()
    fill_diag(c16, 3, "SADD ROUT, R3, ZERO") # Expose nColsi
    fill_diag(c16, 4, "SADD ROUT, R3, ZERO") # Expose (nBlocksColsi + 1)*16
    cycles.append(c16)

    # PC 17
    c17 = create_cycle()
    fill_diag(c17, 3, "SMUL ROUT, ROUT, RCR")
    fill_diag(c17, 4, "SMUL ROUT, ROUT, RCL")
    cycles.append(c17)

    # PC 18
    c18 = create_cycle()
    fill_diag(c18, 1, "SSUB ROUT, 16, RCL")
    fill_diag(c18, 2, "SSUB ROUT, 16, RCR")
    fill_diag(c18, 3, "SADD ROUT, 16, ROUT")
    fill_diag(c18, 4, "SADD ROUT, 16, ROUT")
    cycles.append(c18)

    # PC 19: R1 += offset
    c19 = fill_cycle("SADD R1, R1, ROUT")
    cycles.append(c19)

    # PC 20: reset inner / inc outer
    c20 = create_cycle()
    fill_cell(c20, 0, 0, "SADD R3, ZERO, ZERO")
    fill_cell(c20, 1, 1, "SADD R3, R3, 1")
    cycles.append(c20)

    # PC 21: branch outer loop → PC 4
    c21 = create_cycle()
    fill_cell(c21, 1, 1, "BNE R3, R0, 4")
    cycles.append(c21)

    # PC 22
    c22 = create_cycle()
    fill_cell(c22, 0, 0, "EXIT")
    cycles.append(c22)

    # Export with the path given as argument
    export_csv(cycles, output_path)
