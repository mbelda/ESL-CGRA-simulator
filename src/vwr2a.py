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
        self.lsus = [LSU() for _ in range(CGRA_COLS)]
        self.rcs = [[] for _ in range(CGRA_COLS)]
        for col in range(CGRA_COLS):
            for _ in range(CGRA_ROWS):
                self.rcs[col].append(RC())
        self.mxcus = [MXCU() for _ in range(CGRA_COLS)]
        
    def compileAsm(self, kernel_path, version="", nInstructionsPerCol=64):
        # String buffers
        LCU_instr = [[] for _ in range(CGRA_COLS)]
        LSU_instr = [[] for _ in range(CGRA_COLS)]
        MXCU_instr = [[] for _ in range(CGRA_COLS)]
        RCs_instr = [[[] for _ in range(CGRA_ROWS)] for _ in range(CGRA_COLS)]

        # Load csv file with instructions
        # LCU, LSU, MXCU, RC0, RC1, ..., RCN
        print("Processing file: " + kernel_path + FILENAME_INSTR + version + EXT + "...")
        with open( kernel_path +FILENAME_INSTR+version+EXT, 'r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            # Skip the header (first row)
            next(csv_reader, None)

            # Read every row of the csv
            for col in range(CGRA_COLS):
                for i in range(nInstructionsPerCol):
                    row = next(csv_reader, None)
                    if row is not None:
                        LCU_instr[col].append(row[0])
                        LSU_instr[col].append(row[1])
                        MXCU_instr[col].append(row[2])
                        index = 3
                        for rc in range(CGRA_ROWS):
                            RCs_instr[col][rc].append(row[index])
                            index+=1
                    else:
                        raise ValueError("Expected more rows on the csv. It should have " + str(CGRA_COLS*nInstructionsPerCol) + " rows plus the header.")

        
        # Parse every instruction
        for col in range(CGRA_COLS):
            lcu = self.lcus[col]
            lsu = self.lsus[col]
            rcs = self.rcs[col]
            mxcu = self.mxcus[col]

            for i in range(len(LCU_instr[col])):
                # For LCU
                LCU_inst = LCU_instr[col][i]
                srf_read_idx_lcu, srf_str_idx_lcu = lcu.asmToHex(LCU_inst)
                # For LSU
                LSU_inst = LSU_instr[col][i]
                srf_read_idx_lsu, srf_str_idx_lsu = lsu.asmToHex(LSU_inst)
                # For RCs
                srf_read_idx_rc = [-1 for _ in range(CGRA_ROWS)]
                srf_str_idx_rc = [-1 for _ in range(CGRA_ROWS)]
                vwr_str_rc = [-1 for _ in range(CGRA_ROWS)]
                for row in range(CGRA_ROWS):
                    RCs_inst = RCs_instr[col][row][i]
                    srf_read_idx_rc[row], srf_str_idx_rc[row], vwr_str_rc[row] = rcs[row].asmToHex(RCs_inst)
                
                # ---------------------- Check reads/writes to SRF/VWR ---------------------- 
                # Enable the write to a VWR for each RC
                vwr_row_we = [0 if num == -1 else 1 for num in vwr_str_rc]
                # All the RCs should write to the same VWR in each cycle
                vwr_sel = 0 # Default value
                vwr_str_rc = np.array(vwr_str_rc)
                unique_vwr_str_rc = np.unique(vwr_str_rc)
                if -1 in unique_vwr_str_rc:
                    unique_vwr_str_rc = unique_vwr_str_rc[unique_vwr_str_rc != -1]
                if len(unique_vwr_str_rc) > 1:
                    raise ValueError("Instructions not valid for this cycle of the CGRA. Detected writes from different RCs to different VWRs.")
                if len(unique_vwr_str_rc) > 0 and unique_vwr_str_rc[0] not in {0,1,2}:
                    raise ValueError("Instructions not valid for this cycle of the CGRA. The selected VWR to write is not properly recognised.")
                if len(unique_vwr_str_rc) > 0:
                    vwr_sel = unique_vwr_str_rc[0] # This is already prepared to be 0, 1 or 2, but checked                   

                # Check: Only RC0 should be able to write to SRF
                if np.any(np.array(srf_str_idx_rc[1:]) != -1):
                    raise ValueError("Instructions not valid for this cycle of the CGRA. Only the RC on row 0 can write to the SRF.")
                
                # Check: Only reads to the same SRF register can be made by every unit
                srf_sel = -1 # No one reads
                all_read_srf = srf_read_idx_rc
                all_read_srf.append(srf_read_idx_lcu)
                all_read_srf.append(srf_read_idx_lsu)
                all_read_srf = np.array(all_read_srf)
                unique_vector_read_srf = np.unique(all_read_srf)
                if -1 in unique_vector_read_srf:
                    unique_vector_read_srf = unique_vector_read_srf[unique_vector_read_srf != -1]
                if len(unique_vector_read_srf) > 1:
                    raise ValueError("Instructions not valid for this cycle of the CGRA. Detected reads to different registers of the SRF.")
                if np.any(all_read_srf != -1):
                    srf_sel = (all_read_srf[all_read_srf != -1])[0]

                # Check: Only one write can be done to a register of the SRF
                srf_we = 0 # Default
                all_str_srf = srf_str_idx_rc
                all_str_srf.append(srf_str_idx_lcu)
                all_str_srf.append(srf_str_idx_lsu)
                all_str_srf = np.array(all_str_srf)
                
                str_idx = all_str_srf[all_str_srf != -1]
                if len(str_idx) > 1:
                    raise ValueError("Instructions not valid for this cycle of the CGRA. Detected multiple writes to the SRF.")
                if len(str_idx) > 0:
                    srf_we = 1

                # Check: The reads and writes to the SRF are made to the same register
                if srf_we != 0 and srf_sel != -1 and srf_sel != str_idx[0]:
                    raise ValueError("Instructions not valid for this cycle of the CGRA. Detected reasd and writes to different registers of the SRF.")
                
                # Set who writes to the SRF
                alu_srf_write = 0 # Default
                if srf_str_idx_rc[0] != -1:
                    alu_srf_write = 1 # RC0
                if srf_str_idx_lcu != -1:
                    alu_srf_write = 0 # LCU
                if srf_str_idx_lsu != -1:
                    alu_srf_write = 3 # LSU

                # For MXCU (checks SRF write of itself)
                MXCU_inst = MXCU_instr[col][i]
                mxcu.asmToHex(MXCU_inst, srf_sel, srf_we, alu_srf_write, vwr_row_we, vwr_sel)
        
        # Write instructions to bitstream
        self.instructions_to_header_file(kernel_path)

    def instructions_to_header_file(self, kernel_path):
        with open(kernel_path + 'dsip_bitstream.h', 'w+') as file:
            file.write("#ifndef _DSIP_BITSTREAM_H_\n#define _DSIP_BITSTREAM_H_\n\n#include <stdint.h>\n\n#include \"dsip.h\"\n\n")

            # Write LCU bitstream
            file.write("uint32_t dsip_lcu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
            for col in range(CGRA_COLS): # Think how to control more than one column
                fill_counter = 0
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

            # Write LSU bitstream
            file.write("uint32_t dsip_lsu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
            for col in range(CGRA_COLS): # Think how to control more than one column
                fill_counter = 0
                for i in range(LSU_NUM_CREG):
                    fill_counter+=1
                    if i<IMEM_N_LINES-1:
                        file.write("  {0},\n".format(self.lsus[col].imem.get_word_in_hex(i)))
                    else:
                        file.write("  {0}\n".format(self.lsus[col].imem.get_word_in_hex(i)))
            while fill_counter < IMEM_N_LINES:
                if fill_counter<IMEM_N_LINES-1:
                    file.write("  {0},\n".format(hex(int(self.lsus[col].default_word,2))))
                else:
                    file.write("  {0}\n".format(hex(int(self.lsus[col].default_word,2))))
                fill_counter+=1
            file.write("};\n\n\n")

            # Write MXCU bitstream
            file.write("uint32_t dsip_mxcu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
            for col in range(CGRA_COLS): # Think how to control more than one column
                fill_counter = 0
                for i in range(LSU_NUM_CREG):
                    fill_counter+=1
                    if i<IMEM_N_LINES-1:
                        file.write("  {0},\n".format(self.mxcus[col].imem.get_word_in_hex(i)))
                    else:
                        file.write("  {0}\n".format(self.mxcus[col].imem.get_word_in_hex(i)))
            while fill_counter < IMEM_N_LINES:
                if fill_counter<IMEM_N_LINES-1:
                    file.write("  {0},\n".format(hex(int(self.mxcus[col].default_word,2))))
                else:
                    file.write("  {0}\n".format(hex(int(self.mxcus[col].default_word,2))))
                fill_counter+=1
            file.write("};\n\n\n")

            # Write bitstream of all RCs concatenated
            file.write("uint32_t dsip_rcs_imem_bitstream[4*DSIP_IMEM_SIZE] = {\n")
            for col in range(CGRA_COLS): # Think how to control more than one column
                for row in range(CGRA_ROWS):
                    fill_counter = 0
                    for i in range(LSU_NUM_CREG):
                        fill_counter+=1
                        if i<IMEM_N_LINES-1:
                            file.write("  {0},\n".format(self.rcs[col][row].imem.get_word_in_hex(i)))
                        else:
                            file.write("  {0}\n".format(self.rcs[col][row].imem.get_word_in_hex(i)))
                    while fill_counter < IMEM_N_LINES:
                        if fill_counter<IMEM_N_LINES-1 and row < CGRA_ROWS -1:
                            file.write("  {0},\n".format(hex(int(self.rcs[col][row].default_word,2))))
                        else:
                            file.write("  {0}\n".format(hex(int(self.rcs[col][row].default_word,2))))
                        fill_counter+=1
            file.write("};\n\n\n")

            # Write the endif of the header file
            file.write("#endif // _DSIP_BITSTREAM_H_")
        