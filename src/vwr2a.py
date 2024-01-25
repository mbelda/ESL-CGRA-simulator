import numpy as np
from enum import Enum

from ctypes import c_int32
import csv

from src import *

# CGRA top-level parameters
CGRA_COLS      = 1
CGRA_ROWS      = 4
N_VWR_PER_COL = 3

# Scratchpad memory configuration
SP_NWORDS = 128
SP_NLINES = 64

# Filename
EXT             = ".csv"
FILENAME_INSTR  = "instructions"


#### SPECIALIZED SLOTS: Sub-modules of the VWR2A top module that each perform their own purpos and have their own ISA ######


class CGRA:
    def __init__(self):
        self.lcus = [LCU() for _ in range(CGRA_COLS)]
        
    def compileAsm(self, kernel_path, version=""):
        # String buffers
        LCU_instr = [[] for _ in range(CGRA_COLS)]
        LSU_instr = [[] for _ in range(CGRA_COLS)]
        MXCU_instr = [[] for _ in range(CGRA_COLS)]
        RCs_instr = [[[] for _ in range(CGRA_ROWS)] for _ in range(CGRA_COLS)]

        # Load csv file with instructions
        # LCU, LSU, MXCU, RC0, RC1, RC2, RC3
        with open( kernel_path + "/"+FILENAME_INSTR+version+EXT, 'r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            # Skip the header (first row)
            next(csv_reader, None)

            # Read every row of the csv
            for row in csv_reader:
                for col in range(CGRA_COLS):
                    LCU_instr[col].append(row[0])
                    LSU_instr[col].append(row[1])
                    MXCU_instr[col].append(row[2])
                    index = 3
                    for rc in range(CGRA_ROWS):
                        RCs_instr[col][rc].append(row[index])
                        index+=1
        
        # Parse every instruction
        for col in range(CGRA_COLS):
            lcu = self.lcus[col]
            for i in range(len(LCU_instr[col])):
                LCU_inst = LCU_instr[col][i]
                srf_read_idx_lsu, srf_str_idx_lsu = lcu.asmToHex(LCU_inst)
                # For LSU
                # For RCs
                # Check SRF reads/writes
                # For MXCU
        
        # Write instructions to bitstream
        self.instructions_to_header_file(kernel_path)

    def instructions_to_header_file(self, kernel_path):
        with open(kernel_path + 'dsip_bitstream.h', 'w+') as file:
            file.write("#ifndef _DSIP_BITSTREAM_H_\n#define _DSIP_BITSTREAM_H_\n\n#include <stdint.h>\n\n#include \"dsip.h\"\n\n")

            # Write LCU bitstream
            file.write("uint32_t dsip_lcu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
            fill_counter = 0
            for col in range(CGRA_COLS): # Think how to control more than one column
                for i in range(LCU_NUM_CREG):
                    fill_counter+=1
                    if i<IMEM_N_LINES-1:
                        file.write("  {0},\n".format(self.lcus[col].imem.get_word_in_hex(i)))
                    else:
                        file.write("  {0}\n".format(self.lcus[col].imem.get_word_in_hex(i)))
            while fill_counter < IMEM_N_LINES:
                if fill_counter<IMEM_N_LINES-1:
                    file.write("  {0},\n".format(hex(int(self.lcus[col].default_word,2))))
                else:
                    file.write("  {0}\n".format(hex(int(self.lcus[col].default_word,2))))
                fill_counter+=1
            file.write("};\n\n\n")

            # Write the endif of the header file
            file.write("#endif // _DSIP_BITSTREAM_H_")
        