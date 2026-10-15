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

    # PC 0: Load config values (&A)
    c0 = fill_cycle("LWD R0, 4") 
    cycles.append(c0)

    # PC 1: Load config values (&B)
    c1 = fill_cycle("LWD R1, 4")
    fill_cell(c1, 0, 0, "LWD R2, 4") # NIt Loop
    cycles.append(c1)
    
    # PC 2: Load A values
    c2 = fill_cycle("LWI R3, R0") # R0 = &A
    cycles.append(c2)

    # PC 3: Share &B with Loop Controller RC
    c3 = create_cycle()
    fill_cell(c3, 0, 1, "SSUB ROUT, R1, 4") 
    cycles.append(c3)

    # PC 4: Load B values
    c4 = fill_cycle("LWI ROUT, R1")
    fill_cell(c4, 0, 0, "LWI ROUT, RCR")
    cycles.append(c4)

    # PC 5: Add
    c5 = fill_cycle("SADD ROUT, ROUT, R3")
    cycles.append(c5)

    # PC 6: Store
    c6 = fill_cycle("SWI ROUT, R0")
    cycles.append(c6)

    # PC 7: Update &A
    c7 = fill_cycle("SADD R0, R0, 64")
    cycles.append(c7)

    # PC 8: Update &B and loop iterator
    c8 = fill_cycle("SADD R1, R1, 64")
    fill_cell(c8, 0, 0, "SADD R1, R1, 1") # Loop ++
    cycles.append(c8)

    # PC 9: Loop condition
    c9 = create_cycle()
    fill_cell(c9, 0, 0, "BNE R1, R2, 2")
    cycles.append(c9)

    # PC 10: End
    c10 = create_cycle()
    fill_cell(c10, 0, 0, "EXIT")
    cycles.append(c10)

    # Export with the path given as argument
    export_csv(cycles, output_path)
