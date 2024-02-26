# Filename
EXT             = ".csv"
FILENAME_INSTR  = "instructions"

import numpy as np
from enum import Enum

from ctypes import c_int32
import csv

from .vwr2a import CGRA, CGRA_ROWS, CGRA_COLS
from .spm import *
from .imem import IMEM_N_LINES
from .lcu import LCU_NUM_CREG
from .lsu import LSU_NUM_CREG
from .mxcu import MXCU_NUM_CREG
from .rc import RC_NUM_CREG

class SIMULATOR:
    def __init__(self):
        self.vwr2a = CGRA()
    
    def kernel_config(self, column_usage, kernel_nInstr, imem_add_start, srf_spm_addres, kernel_number):
        self.vwr2a.kernel_config(column_usage, kernel_nInstr, imem_add_start, srf_spm_addres, kernel_number)

    def setSPMLine(self, nline, vector):
        self.vwr2a.setSPMLine(nline, vector)
    
    def loadSPMData(self, data):
        self.vwr2a.loadSPMData(data)

    def run(self, nKernel, display_ops=[[] for _ in range(CGRA_ROWS + 4)]): # +4 -> (LCU, LSU, MXCU, SRF)
        # Decode the kernel number of instructions and which ones they are
        n_instr, imem_start_addr, col_usage, srf_spm_bank = self.vwr2a.kmem.imem.get_params(nKernel)
        n_instr+=1 # Add one
       
        # Control the columns used
        if col_usage == 0:
            ini_col = 0
            end_col = 0
            col = 0
        elif col_usage == 2:
            ini_col = 1
            end_col = 1
            col = 1
        else: # col_usage == 1
            # Both
            ini_col = 0
            end_col = 1
        
        # Initialize the index of the SRF values on the SPM on R7 of the LSU
        for col in range(ini_col, end_col+1):
            self.vwr2a.lsu[col].regs[7] = srf_spm_bank
        
        # Execute each pc instruction on each used column
        pc = imem_start_addr # The pc is the same for both columns because is the same kernel

        branch = [-1 for _ in range(ini_col, end_col+1)]
        exit = [-1 for _ in range(ini_col, end_col+1)]
        
        while pc < imem_start_addr+n_instr and not exit:
            for col in range(ini_col, end_col+1):
                self.vwr2a.lsus[col].run(pc) # Check if they need anything from the others
                self.vwr2a.mxcus[col].run(pc)
                for rc in range(CGRA_ROWS):
                    self.vwr2a.rcs[col][rc].run(pc)
                # Last the LCU because it might need the ALU flags of the RCs
                branch[col], exit[col] = self.vwr2a.lcus[col].run(pc) # Is a branch is taken, returns the inm
            # Update pc
            pc+=1
            for col in range(ini_col, end_col+1):
                if branch[col] != -1:
                    pc = imem_start_addr + branch[col]
                if exit:
                    #TODO
                    pass
    
    def compileAsm(self, kernel_path, version="", nInstrPerCol=0, colUsage=[False,False], imem_start_addr=0):
        # String buffers
        LCU_instr = [[] for _ in range(CGRA_COLS)]
        LSU_instr = [[] for _ in range(CGRA_COLS)]
        MXCU_instr = [[] for _ in range(CGRA_COLS)]
        RCs_instr = [[[] for _ in range(CGRA_ROWS)] for _ in range(CGRA_COLS)]

        # Load csv file with instructions
        # LCU, LSU, MXCU, RC0, RC1, ..., RCN
        print("Processing file: " + kernel_path + FILENAME_INSTR + version + EXT + "...")
        with open( kernel_path +FILENAME_INSTR+version+EXT, 'r') as file:

            # Control the used columns
            assert(len(colUsage) == 2), "The column usage must have the structure [True/Flase, True/False]"
                
            if colUsage[0]: ini_col = 0
            else: ini_col = 1

            if colUsage[1]: end_col = 1
            else: end_col = 0
            
            nUsedCols = (end_col - ini_col +1)
            assert(nUsedCols > 0), "At least one column must be used"

            # Create a CSV reader object
            csv_reader = csv.reader(file)
            # Skip the header (first row)
            next(csv_reader, None)

            # For each used column read the number of instructions
            for col in range(ini_col, end_col+1):
                instr_cont = 0
                while instr_cont < nInstrPerCol:
                    try:
                        row = next(csv_reader, None)
                        LCU_instr[col].append(row[0])
                        LSU_instr[col].append(row[1])
                        MXCU_instr[col].append(row[2])
                    
                        index = 3
                        for rc in range(CGRA_ROWS):
                            RCs_instr[col][rc].append(row[index])
                            index+=1
                    except:
                        raise Exception("CSV instruction structure is not appropiate. Expected: LCU_instr, LSU_instr, MXCU_instr, RC0_instr, ..., RC" + str(CGRA_ROWS -1) + "_instr. It should have " + str(nUsedCols*nInstrPerCol) + " rows plus the header.")
                    instr_cont+=1
        
        # Parse every instruction
        imem_addr = imem_start_addr
        for col in range(ini_col, end_col+1):
            lcu = self.vwr2a.lcus[col]
            lsu = self.vwr2a.lsus[col]
            rcs = self.vwr2a.rcs[col]
            mxcu = self.vwr2a.mxcus[col]
            srf = self.vwr2a.srfs[col]

            for i in range(len(LCU_instr[col])):
                # For LCU
                LCU_inst = LCU_instr[col][i]
                srf_read_idx_lcu, srf_str_idx_lcu, hex_word = lcu.asmToHex(LCU_inst)
                self.vwr2a.imem.lcu_imem[imem_addr] = hex_word
                # For LSU
                LSU_inst = LSU_instr[col][i]
                srf_read_idx_lsu, srf_str_idx_lsu, hex_word = lsu.asmToHex(LSU_inst)
                self.vwr2a.imem.lsu_imem[imem_addr] = hex_word
                # For RCs
                srf_read_idx_rc = [-1 for _ in range(CGRA_ROWS)]
                srf_str_idx_rc = [-1 for _ in range(CGRA_ROWS)]
                vwr_str_rc = [-1 for _ in range(CGRA_ROWS)]
                for row in range(CGRA_ROWS):
                    RCs_inst = RCs_instr[col][row][i]
                    srf_read_idx_rc[row], srf_str_idx_rc[row], vwr_str_rc[row], hex_word = rcs[row].asmToHex(RCs_inst)
                    self.vwr2a.imem.rcs_imem[row][imem_addr] = hex_word
                
                # Check SRF reads/writes
                srf_sel, srf_we, alu_srf_write = srf.checkReadsWrites(srf_read_idx_lcu, srf_read_idx_lsu, srf_read_idx_rc, srf_str_idx_lcu, srf_str_idx_lsu, srf_str_idx_rc)
                
                # Check vwr reads/writes
                # Enable the write to a VWR for each RC
                vwr_row_we = [0 if num == -1 else 1 for num in vwr_str_rc]
                # All the RCs should write to the same VWR in each cycle
                vwr_sel = 0 # Default value
                vwr_str_rc = np.array(vwr_str_rc)
                unique_vwr_str_rc = np.unique(vwr_str_rc)
                if -1 in unique_vwr_str_rc:
                    unique_vwr_str_rc = unique_vwr_str_rc[unique_vwr_str_rc != -1]
                if len(unique_vwr_str_rc) > 1:
                    raise Exception("Instructions not valid for this cycle of the CGRA. Detected writes from different RCs to different VWRs.")
                if len(unique_vwr_str_rc) > 0 and unique_vwr_str_rc[0] not in {0,1,2}:
                    raise Exception("Instructions not valid for this cycle of the CGRA. The selected VWR to write is not properly recognised.")
                if len(unique_vwr_str_rc) > 0:
                    vwr_sel = unique_vwr_str_rc[0] # This is already prepared to be 0, 1 or 2, but checked                   
                
                # For MXCU (checks SRF write of itself)
                MXCU_inst = MXCU_instr[col][i]
                hex_word = mxcu.asmToHex(MXCU_inst, srf_sel, srf_we, alu_srf_write, vwr_row_we, vwr_sel)
                self.vwr2a.imem.mxcu_imem[imem_addr] = hex_word

                imem_addr+=1
        
        # Write instructions to bitstream
        self.instructions_to_header_file(kernel_path)

    def instructions_to_header_file(self, kernel_path):
        with open(kernel_path + 'dsip_bitstream.h', 'w+') as file:
            file.write("#ifndef _DSIP_BITSTREAM_H_\n#define _DSIP_BITSTREAM_H_\n\n#include <stdint.h>\n\n#include \"dsip.h\"\n\n")

            # Write LCU bitstream
            file.write("uint32_t dsip_lcu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
            for i in range(IMEM_N_LINES):
                if i<IMEM_N_LINES-1:
                    file.write("  {0},\n".format(self.vwr2a.imem.lcu_imem[i]))
                else:
                    file.write("  {0}\n".format(self.vwr2a.imem.lcu_imem[i]))
            file.write("};\n\n\n")

            # Write LSU bitstream
            file.write("uint32_t dsip_lsu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
            for i in range(IMEM_N_LINES):
                if i<IMEM_N_LINES-1:
                    file.write("  {0},\n".format(self.vwr2a.imem.lsu_imem[i]))
                else:
                    file.write("  {0}\n".format(self.vwr2a.imem.lsu_imem[i]))
            file.write("};\n\n\n")

            # Write MXCU bitstream
            file.write("uint32_t dsip_mxcu_imem_bitstream[DSIP_IMEM_SIZE] = {\n")
            for i in range(IMEM_N_LINES):
                if i<IMEM_N_LINES-1:
                    file.write("  {0},\n".format(self.vwr2a.imem.mxcu_imem[i]))
                else:
                    file.write("  {0}\n".format(self.vwr2a.imem.mxcu_imem[i]))
            file.write("};\n\n\n")

            # Write bitstream of all RCs concatenated
            file.write("uint32_t dsip_rcs_imem_bitstream[4*DSIP_IMEM_SIZE] = {\n")
            for row in range(CGRA_ROWS): # For each RC
                for i in range(IMEM_N_LINES):
                    if i < CGRA_ROWS*IMEM_N_LINES-1:
                        file.write("  {0},\n".format(self.vwr2a.imem.rcs_imem[row][i]))
                    else:
                        file.write("  {0}\n".format(self.vwr2a.imem.rcs_imem[row][i]))
            file.write("};\n\n\n")

            # Write the endif of the header file
            file.write("#endif // _DSIP_BITSTREAM_H_")
        