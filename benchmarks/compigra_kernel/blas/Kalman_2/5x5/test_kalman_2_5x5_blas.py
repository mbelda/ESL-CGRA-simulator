import random
from cgra import *
from kernels import *

# Global variables
CGRA_N_ROWS = 5
CGRA_N_COLS = 5
SIZE = 60

# Adress
first_addr = 20000

# Benchmark
kernel_name = f"benchmarks/compigra_blas_paper/blas/Kalman_2/{CGRA_N_ROWS}x{CGRA_N_COLS}/"
version = f"_out_{CGRA_N_COLS}_I{SIZE}_J{SIZE}_K{SIZE}"

# ------------------------------------
#           FUNCTIONS
# ------------------------------------
def configMemory(A, Q, AP, NI):
    # Clear memory values
    kernel_clear_memory(kernel_name, version=version)
    # Config values
    # ----------------------            
    first_addr_A = first_addr
    first_addr_Q = first_addr_A + NI*NI*4
    first_addr_AP = first_addr_Q + NI*NI*4
    first_addr_AT = first_addr_AP + NI*NI*4
    first_addr_APA = first_addr_AT + NI*NI*4
    first_addr_P = first_addr_APA + NI*NI*4

    config_vals = [[] for i in range(CGRA_N_COLS)]

    # void Kalman_2(int A[NI][NI], int Q[NI][NI], int AP[NI][NI], int AT[NI][NI], int APA[NI][NI], int P[NI][NI])
    # 0 : first_addr_A
    # 1 : first_addr_Q
    # 2 : first_addr_AP
    # 3 : first_addr_AT
    # 4 : first_addr_APA
    # 5 : first_addr_P


    config_vals[0] = [first_addr_P] # 5
    config_vals[1] = [first_addr_Q, first_addr_AT] # 1, 3
    config_vals[2] = [first_addr_APA, first_addr_APA] # 4, 4
    config_vals[3] = [first_addr_AP] # 2
    config_vals[4] = [first_addr_A, first_addr_AT ] # 0, 3


    addr_config_loads = [0 for i in range(CGRA_N_COLS)]
    for i in range(CGRA_N_COLS):
        kernel_add_memory_region(kernel_name, addr_config_loads[i], config_vals[i], version=version)
        if i < CGRA_N_COLS -1:
            addr_config_loads[i+1] = addr_config_loads[i] + len(config_vals[i])*4
            
    # Load data
    kernel_add_memory_region(kernel_name, first_addr_A, A, version=version)
    kernel_add_memory_region(kernel_name, first_addr_Q, Q, version=version)
    kernel_add_memory_region(kernel_name, first_addr_AP, AP, version=version)

    # Config data address for direct loads
    return addr_config_loads

def printAsMatrix(array, rows, cols):
    for i in range(rows):
        print(array[i * cols:(i + 1) * cols])

def runKernel(load_addrs, max_it=1000, pr=["ROUT","INST"], printVal=1):
    # Run kernel
    run(kernel_name, pr=pr, load_addrs=load_addrs, version=version, limit=max_it, printVal=printVal)

def getResult(first_addr, length):
    result = [0 for _ in range(length)]
    with open( kernel_name + "/memory_out"+version+".csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            try:
                if(int(row[0]) >= first_addr) and (int(row[0]) < first_addr + length*4):
                    result[int((int(row[0]) - first_addr)/4)] = int(row[1])
            except ValueError:
                print("Error: Values in memory_out CSV file are not integers.")
    return result

# --------------------------------------------
#               DATA
# --------------------------------------------
data = np.load(kernel_name + f"data/data_{SIZE}.npz")

A = data["A"]
Q = data["Q"]
AP = data["AP"]
NI = int(data["NI"])

# Expected results
AT_expected = data["AT_expected"]
APA_expected = data["APA_expected"]
P_expected = data["P_expected"]

print(f"Testing Kalman_2 sizes : {NI}")

load_addrs = configMemory(A, Q, AP, NI)

runKernel(load_addrs, max_it=2000000000, printVal=0)
#estimatedConfigCycles(kernel_name, version)

# Get result from CGRA
first_addr_AT_res = first_addr + NI*NI*4*3
AT_result = getResult(first_addr_AT_res, NI*NI)

first_addr_APA_res = first_addr + NI*NI*4*4
APA_result = getResult(first_addr_APA_res, NI*NI)

first_addr_P_res = first_addr + NI*NI*4*5
P_result = getResult(first_addr_P_res, NI*NI)

# Check result correctness
print("Check AT result:")
errors = 0
err_idx = []
for i in range(len(AT_expected)):
    if AT_expected[i] != AT_result[i]:
        errors += 1
        err_idx.append(i)
if errors > 0:
    print("Err: " + str(errors))
    print("Expected: ")
    printAsMatrix(AT_expected, 1, NI)
    print("CGRA: ")
    printAsMatrix(AT_result, 1, NI)
    print("Errors are: Exp : CGRA")
    for i in err_idx:
        print(f"Idx[{i}] {AT_expected[i]} : {AT_result[i]}")
else:
    print("OK")

print("Check APA result:")
errors = 0
err_idx = []
for i in range(len(APA_expected)):
    if APA_expected[i] != APA_result[i]:
        errors += 1
        err_idx.append(i)
if errors > 0:
    print("Err: " + str(errors))
    print("Expected: ")
    printAsMatrix(APA_expected, NI, NI)
    print("CGRA: ")
    printAsMatrix(APA_result, NI, NI)
    print("Errors are: Exp : CGRA")
    for i in err_idx:
        row = int(i/NI)
        col = i%NI
        print(f"Idx[{row}][{col}] {APA_expected[i]} : {APA_result[i]}")
else:
    print("OK")

print("Check P result:")
errors = 0
err_idx = []
for i in range(len(P_expected)):
    if P_expected[i] != P_result[i]:
        errors += 1
        err_idx.append(i)
if errors > 0:
    print("Err: " + str(errors))
    print("Expected: ")
    printAsMatrix(P_expected, NI, NI)
    print("CGRA: ")
    printAsMatrix(P_result, NI, NI)
    print("Errors are: Exp : CGRA")
    for i in err_idx:
        row = int(i/NI)
        col = i%NI
        print(f"Idx[{row}][{col}] {P_expected[i]} : {P_result[i]}")
else:
    print("OK")
