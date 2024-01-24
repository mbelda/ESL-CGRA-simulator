import csv
import re
from ctypes import c_int32

EXT             = ".csv"
FILENAME_INSTR  = "instructions"
CGRA_COLS = 1
CGRA_ROWS = 4

LCU_instr=[]
LSU_instr=[]
RCs_instr=[]


def checkForSRFReadWrite(instructions):
    pattern = r'SRF\((\d+)\)'
    srf_index = -1
    for i in range(len(instructions)):
        match = re.search(pattern, instructions[i])
        if match:
            number = int(match.group(1))
            if srf_index != -1 and srf_index != number:
                raise ValueError("More than one instruction per cycle access the SRF on different indexes")
            srf_index = number
    return srf_index

def asmToHex():
    for col in range(CGRA_COLS):
        for i in range(len(LCU_instr)):
            LCU_inst = LCU_instr[col][i]
            lcu.asmToHex(LCU_inst)
            LSU_inst = LSU_instr[col][i]
            i_srf_read = checkForSRFReadWrite([LCU_inst, LSU_inst] +  [RCs_instr[col][row][i] for row in range(CGRA_ROWS)])
            
            lsu.asmToHex(LSU_inst)
            for row in range(CGRA_ROWS):
                RC_inst = RCs_instr[col][row][i]
                i_srf = checkForSRF(RC_inst)


def compileAsm(instr, version=""):
    
    for col in range(CGRA_COLS):
        LCU_instr[col] = []
        LSU_instr[col] = []
        RCs_instr[col] = []
        for row in range(CGRA_ROWS):
            RCs_instr[col].append([])

    # Load csv file with instructions
    # LCU, LSU, RC0, RC1, RC2, RC3
    with open( instr + "/"+FILENAME_INSTR+version+EXT, 'r') as f:
        for row in csv.reader(f):
            for col in range(CGRA_COLS):
                LCU_instr[col].append(row[0])
                LSU_instr[col].append(row[1])
                index = 2
                for rc in range(CGRA_ROWS):
                    RCs_instr[col][rc].append(row[index])
                    index+=1
    
    asmToHex()



                    