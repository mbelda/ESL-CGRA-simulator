"""lsu.py: Data structures and objects emulating the Load Store Unit of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
from enum import Enum
from ctypes import c_int32
import re

from .srf import SRF_N_REGS

# Local data register (DREG) sizes of specialized slots
LSU_NUM_DREG = 8

# Configuration register (CREG) / instruction memory sizes of specialized slots
LSU_NUM_CREG = 64

# Widths of instructions of each specialized slot in bits
LSU_IMEM_WIDTH = 20

# LSU IMEM word decoding
class LSU_ALU_OPS(int, Enum):
    '''LSU ALU operation codes'''
    LAND = 0
    LOR = 1
    LXOR = 2
    SADD = 3
    SSUB = 4
    SLL = 5
    SRL = 6
    BITREV = 7

class LSU_DEST_REGS(int, Enum):
    '''Available registers to store ALU result'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    SRF = 8

class LSU_MUX_SEL(int, Enum):
    '''Input A to LSU ALU'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    SRF = 8
    ZERO = 9
    ONE = 10
    TWO = 11

class LSU_MEM_OP(int, Enum):
    '''Select whether the LSU is interfacing with the SPM or shuffling VWR data'''
    NOP = 0
    LOAD = 1
    STORE = 2
    SHUFFLE = 3

class LSU_VWR_SEL(int, Enum):
    '''When the LSU OP MODE is in LOAD/STORE, choose which element to load or store from'''
    VWR_A = 0
    VWR_B = 1
    VWR_C = 2
    SRF = 3
    
class SHUFFLE_SEL(int, Enum):
    '''When the LSU OP MODE is in SHUFFLE, choose how to shuffle VWRs A and B into VWR C'''
    INTERLEAVE_UPPER = 0
    INTERLEAVE_LOWER = 1
    EVEN_INDICES = 2
    ODD_INDICES = 3
    CONCAT_BITREV_UPPER = 4
    CONCAT_BITREV_LOWER = 5
    CONCAT_SLICE_CIRCULAR_SHIFT_UPPER = 6
    CONCAT_SLICE_CIRCULAR_SHIFT_LOWER = 7
    
# LOAD STORE UNIT (LSU) #

class LSU_IMEM:
    '''Instruction memory of the Load Store Unit'''
    def __init__(self):
        self.IMEM = np.zeros(LSU_NUM_CREG,dtype="S{0}".format(LSU_IMEM_WIDTH))
        # Initialize kernel memory with default instruction
        default_word = LSU_IMEM_WORD()
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = default_word.get_word()
    
    def set_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary imem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=LSU_IMEM_WIDTH)
    
    def set_params(self, rf_wsel=0, rf_we=0, alu_op=LSU_ALU_OPS.LAND, muxb_sel=LSU_MUX_SEL.ZERO, muxa_sel=LSU_MUX_SEL.ZERO, vwr_sel_shuf_op=LSU_VWR_SEL.VWR_A, mem_op=LSU_MEM_OP.NOP, pos=0):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See LSU_IMEM_WORD initializer for implementation details.
        '''
        imem_word = LSU_IMEM_WORD(rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel, vwr_sel_shuf_op, mem_op)
        self.IMEM[pos] = imem_word.get_word()
    
    def get_instruction_info(self, pos):
        '''Print the human-readable instructions of the instruction at position pos in the instruction memory'''
        imem_word = LSU_IMEM_WORD()
        imem_word.set_word(self.IMEM[pos])
        rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel, vwr_sel_shuf_op, mem_op = imem_word.decode_word()
        
        # See if we are performing a load/store or a shuffle
        for op in LSU_MEM_OP:
            if op.value == mem_op:
                lsu_mode = op.name
        
        # If we are loading/storing ...
        if ((lsu_mode == LSU_MEM_OP.STORE.name)|(lsu_mode == LSU_MEM_OP.LOAD.name)):
            #... which register are we using?
            for reg in LSU_VWR_SEL:
                if reg.value == vwr_sel_shuf_op:
                    register = reg.name
            if (lsu_mode == LSU_MEM_OP.LOAD.name):
                preposition1 = "from"
                preposition2 = "to"
            else:
                preposition1 = "to"
                preposition2 = "from"
            print("Performing {0} {1} SPM {2} {3}".format(lsu_mode, preposition1, preposition2, register))
        # If we are shuffling VWRs A and B into C ...
        elif (lsu_mode == LSU_MEM_OP.SHUFFLE.name):
            #... which one are we using?
            for shuf in SHUFFLE_SEL:
                if shuf.value == vwr_sel_shuf_op:
                    shuffle_type = shuf.name
            print("Shuffling VWR A and B data into VWR C using operation {0}".format(shuffle_type))
        else: # NOP
            print("No loading, storing, or shuffling taking place")
        
        for op in LSU_ALU_OPS:
            if op.value == alu_op:
                alu_opcode = op.name
        for sel in LSU_MUX_SEL:
            if sel.value == muxa_sel:
                muxa_res = sel.name
        for sel in LSU_MUX_SEL:
            if sel.value == muxb_sel:
                muxb_res = sel.name
        
        print("Performing ALU operation {0} between operands {1} and {2}".format(alu_opcode, muxa_res, muxb_res))
        
        if rf_we == 1:
            print("Writing ALU result to LSU register {0}".format(rf_wsel))
        else:
            print("No LSU registers are being written")
        
        
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the LSU config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class LSU_IMEM_WORD:
    def __init__(self, hex_word=None, rf_wsel=0, rf_we=0, alu_op=LSU_ALU_OPS.LAND, muxb_sel=LSU_MUX_SEL.ZERO, muxa_sel=LSU_MUX_SEL.ZERO, vwr_sel_shuf_op=LSU_VWR_SEL.VWR_A, mem_op=LSU_MEM_OP.NOP):
        '''Generate a binary lsu instruction word from its configuration paramerers:
        
           -   rf_wsel: Select one of eight LSU registers to write to
           -   rf_we: Enable writing to aforementioned register
           -   alu_op: Perform one of the ALU operations listed in the LSU_ALU_OPS enum
           -   muxb_sel: Select input B to ALU (see LSU_MUX_SEL enum for options)
           -   muxa_sel: Select input A to ALU (see LSU_MUX_SEL enum for options)
           -   vwr_sel_shuf_op: Decide which VWR to load/store to (LSU_VWR_SEL) or which shuffle operation to perform (SHUFFLE_SEL)
           -   mem_op: Decide whether to use LSU for SPM communication or data shuffling (see LSU_MEM_OP enum for options)
        
        '''
        if hex_word == None:
            self.rf_wsel = np.binary_repr(rf_wsel, width=3)
            self.rf_we = np.binary_repr(rf_we,width=1)
            self.alu_op = np.binary_repr(alu_op,3)
            self.muxb_sel = np.binary_repr(muxb_sel,4)
            self.muxa_sel = np.binary_repr(muxa_sel,4)
            self.vwr_sel_shuf_op = np.binary_repr(vwr_sel_shuf_op,3)
            self.mem_op = np.binary_repr(mem_op,2)
            self.word = "".join((self.mem_op,self.vwr_sel_shuf_op,self.muxa_sel,self.muxb_sel,self.alu_op,self.rf_we,self.rf_wsel))
        else:
            decimal_int = int(hex_word, 16)
            binary_string = bin(decimal_int)[2:]  # Removing the '0b' prefix
            self.rf_wsel = binary_string[:3]
            self.rf_we = binary_string[3:4]
            self.alu_op = binary_string[4:7]
            self.muxb_sel = binary_string[7:11]
            self.muxa_sel = binary_string[11:15]
            self.vwr_sel_shuf_op = binary_string[15:18]
            self.mem_op = binary_string[18:20]
            self.word = binary_string
    
    def get_word(self):
        return self.word
    
    def get_word_in_hex(self):
        '''Get the hexadecimal representation of the word at index pos in the LSU config IMEM'''
        return(hex(int(self.word, 2)))
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.rf_wsel = word[17:]
        self.rf_we = word[16:17]
        self.alu_op = word[13:16]
        self.muxb_sel = word[9:13]
        self.muxa_sel = word[5:9]
        self.vwr_sel_shuf_op = word[2:5]
        self.mem_op = word[0:2]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''
        rf_wsel = int(self.rf_wsel,2)
        rf_we = int(self.rf_we,2)
        alu_op = int(self.alu_op,2)
        muxb_sel = int(self.muxb_sel,2)
        muxa_sel = int(self.muxa_sel,2)
        vwr_sel_shuf_op = int(self.vwr_sel_shuf_op,2)
        mem_op = int(self.mem_op,2)
        
        
        return rf_wsel, rf_we, alu_op, muxb_sel, muxa_sel, vwr_sel_shuf_op, mem_op
    

class LSU:
    lsu_arith_ops   = { 'SADD','SSUB','SLL','SRL','LAND','LOR','LXOR' }
    lsu_nop_ops     = { 'NOP' }
    lsu_mem_ops     = { 'LD.VWR','ST.VWR' }
    lsu_shuf_ops    = { 'SH.IL.UP','SH.IL.LO','SH.EVEN','SH.ODD','SH.BRE.UP','SH.BRE.LO','SH.CSHIFT.UP','SH.CSHIFT.LO' }

    def __init__(self):
        self.regs       = [0 for _ in range(LSU_NUM_DREG)]
        self.imem       = LSU_IMEM()
        self.nInstr     = 0
        self.default_word = LSU_IMEM_WORD().get_word()
    
    # def sadd( val1, val2 ):
    #     return c_int32( val1 + val2 ).value

    # def ssub( val1, val2 ):
    #     return c_int32( val1 - val2 ).value

    # def sll( val1, val2 ):
    #     return c_int32(val1 << val2).value

    # def srl( val1, val2 ):
    #     interm_result = (c_int32(val1).value & MAX_32b)
    #     return c_int32(interm_result >> val2).value

    # def lor( val1, val2 ):
    #     return c_int32( val1 | val2).value

    # def land( val1, val2 ):
    #     return c_int32( val1 & val2).value

    # def lxor( val1, val2 ):
    #     return c_int32( val1 ^ val2).value

    # def nop(self):
    #     pass # Intentional

    # def load_vwr(self):
    #     pass

    # def store_vwr(self):
    #     pass

    # def shilup(self):
    #     pass
    
    # def shillo(self):
    #     pass

    # def sheven(self):
    #     pass

    # def shodd(self):
    #     pass

    # def shbreup(self):
    #     pass

    # def shbrelo(self):
    #     pass

    # def shcshiftup(self):
    #     pass

    # def shcshiftlo(self):
    #     pass

    def run(self, pc):
        print(self.__class__.__name__ + ": " + self.imem.get_word_in_hex(pc))
        pass

    def parseDestArith(self, rd, instr):
        # Define the regular expression pattern
        r_pattern = re.compile(r'^R(\d+)$')
        srf_pattern = re.compile(r'^SRF\((\d+)\)$')

        # Check if the input matches the 'R' pattern
        r_match = r_pattern.match(rd)
        if r_match:
            ret = None
            try:
                ret = LSU_DEST_REGS[rd]
            except:
                raise ValueError("Instruction not valid for LSU: " + instr + ". The accessed register must be betwwen 0 and " + str(len(self.regs) -1) + ".")
            return ret, -1

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rd)
        if srf_match:
            return LSU_DEST_REGS["SRF"], int(srf_match.group(1))

        return None, -1

    def parseMuxArith(self, rs, instr):
        # Define the regular expression pattern
        r_pattern = re.compile(r'^R(\d+)$')
        srf_pattern = re.compile(r'^SRF\((\d+)\)$')
        zero_pattern = re.compile(r'^ZERO$')
        one_pattern = re.compile(r'^ONE$')
        two_pattern = re.compile(r'^TWO$')

        # Check if the input matches the 'R' pattern
        r_match = r_pattern.match(rs)
        if r_match:
            ret = None
            try:
                ret = LSU_MUX_SEL[rs]
            except:
                raise ValueError("Instruction not valid for LSU: " + instr + ". The accessed register must be betwwen 0 and " + str(len(self.regs) -1) + ".")
            return ret, -1

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rs)
        if srf_match:
            return LSU_MUX_SEL["SRF"], int(srf_match.group(1))
        
        # Check if the input matches the 'ZERO' pattern
        zero_match = zero_pattern.match(rs)
        if zero_match:
            return LSU_MUX_SEL[rs], -1
        
        # Check if the input matches the 'ONE' pattern
        one_match = one_pattern.match(rs)
        if one_match:
            return LSU_MUX_SEL[rs], -1
        
        # Check if the input matches the 'TWO' pattern
        two_match = two_pattern.match(rs)
        if two_match:
            return LSU_MUX_SEL[rs], -1

        return None, -1
    
    def parseDestMem(self, rd):
        # Define the regular expression pattern
        srf_pattern = re.compile(r'^SRF$')
        vwr_a = re.compile(r'^VWR_A$')
        vwr_b = re.compile(r'^VWR_B$')
        vwr_c = re.compile(r'^VWR_C$')

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rd)
        if srf_match:
            return LSU_VWR_SEL["SRF"]

        # Check if the input matches the 'VWR_A' pattern
        vwra_match = vwr_a.match(rd)
        if vwra_match:
            return LSU_VWR_SEL["VWR_A"]
        
        # Check if the input matches the 'VWR_B' pattern
        vwrb_match = vwr_b.match(rd)
        if vwrb_match:
            return LSU_VWR_SEL["VWR_B"]
        
        # Check if the input matches the 'VWR_C' pattern
        vwrc_match = vwr_c.match(rd)
        if vwrc_match:
            return LSU_VWR_SEL["VWR_C"]

        return None

    def asmToHex(self, instr):

        instructions = instr.split("/")
        split_instr = [itr.replace(",", " ") for itr in instructions]
        try:
            arith_instr = [word for word in split_instr[0].split(" ") if word]
            mem_instr = [word for word in split_instr[1].split(" ") if word]
        except:
            raise ValueError("Instruction not valid for LSU: " + instr + ". Expected 2 instructions: arith/mem.")

        # Arithmetic instruction
        try:
            op      = arith_instr[0]
        except:
            op      = arith_instr

        if op in self.lsu_arith_ops:
            alu_op = LSU_ALU_OPS[op]
            # Expect 3 operands: rd/srf, rs/srf/zero/one, rt/srf/zero/one
            try:
                rd = arith_instr[1]
                rs = arith_instr[2]
                rt = arith_instr[3]
            except:
                raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". Expected 3 operands.")
            dest, srf_str_index = self.parseDestArith(rd, instr)
            muxA, srf_muxA_index = self.parseMuxArith(rs, instr)
            muxB, srf_read_index = self.parseMuxArith(rt, instr)

            if srf_read_index > SRF_N_REGS or srf_muxA_index > SRF_N_REGS or srf_str_index > SRF_N_REGS:
                raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". The accessed SRF must be between 0 and " + str(SRF_N_REGS -1) + ".")

            if dest == None:
                raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". Expected another format for first operand (dest).")
            
            if muxB == None:
                raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". Expected another format for the second operand (muxB).")

            if muxA == None:
                raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". Expected another format for the third operand (muxA).")
            
            if srf_muxA_index != -1:
                if srf_read_index != -1 and srf_muxA_index != srf_read_index:
                    raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". Expected only reads/writes to the same reg of the SRF.") 
                srf_read_index = srf_muxA_index

            if srf_str_index != -1 and srf_read_index != -1 and srf_str_index != srf_read_index:
                raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". Expected only reads/writes to the same reg of the SRF.")

            if srf_str_index == -1: # Writting on a local reg
                rf_we = 1
                rf_wsel = dest
            else:
                rf_wsel = 0
                rf_we = 0
        else:
            raise ValueError("Instruction not valid for LSU: " + instructions[0] + ". Arithmetic operation not recognised.")
        
        # Memory instruction
        try:
            op      = mem_instr[0]
        except:
            op      = mem_instr

        if op in self.lsu_nop_ops:
            mem_op = LSU_MEM_OP[op]
            vwr_sel_shuf_op = 0

        elif op in self.lsu_mem_ops:
            # Expect 1 operand: VWRA/VWRB/VWRC/SRF
            try:
                rd = mem_instr[1]
            except:
                raise ValueError("Instruction not valid for LSU: " + instructions[1] + ". Expected 1 operand.")
            vwr_sel_shuf_op = self.parseDestMem(rd)
            if vwr_sel_shuf_op == None:
                raise ValueError("Instruction not valid for LSU: " + instructions[1] + ". Memory destination not recognized.")
            if op == "LD.VWR":
                mem_op = LSU_MEM_OP["LOAD"]
            else:
                mem_op = LSU_MEM_OP["STORE"]
        
        elif op in self.lsu_shuf_ops:
            mem_op = LSU_MEM_OP["SHUFFLE"]
            if op == "SH.IL.UP":
                vwr_sel_shuf_op = SHUFFLE_SEL["INTERLEAVE_UPPER"]
            elif op == "SH.IL.LO":
                vwr_sel_shuf_op = SHUFFLE_SEL["INTERLEAVE_LOWER"]
            elif op == "SH.EVEN":
                vwr_sel_shuf_op = SHUFFLE_SEL["EVEN_INDICES"]
            elif op == "SH.ODD":
                vwr_sel_shuf_op = SHUFFLE_SEL["ODD_INDICES"]
            elif op == "SH.BRE.UP":
                vwr_sel_shuf_op = SHUFFLE_SEL["CONCAT_BITREV_UPPER"]
            elif op == "SH.BRE.LO":
                vwr_sel_shuf_op = SHUFFLE_SEL["CONCAT_BITREV_LOWER"]
            elif op == "SH.CSHIFT.UP":
                vwr_sel_shuf_op = SHUFFLE_SEL["CONCAT_SLICE_CIRCULAR_SHIFT_UPPER"]
            elif op == "SH.CSHIFT.LO":
                vwr_sel_shuf_op = SHUFFLE_SEL["CONCAT_SLICE_CIRCULAR_SHIFT_LOWER"]
            else:
                raise ValueError("Instruction not valid for LSU: " + instructions[1] + ". Shuffle operation not recognized.")

        else:
            raise ValueError("Instruction not valid for LSU: " + instructions[1] + ". Memory operation not recognised.")    

        # Add hexadecimal instruction
        #self.imem.set_params(mem_op=mem_op, vwr_sel_shuf_op=vwr_sel_shuf_op, rf_wsel=rf_wsel, rf_we=rf_we, alu_op=alu_op, muxb_sel=muxB, muxa_sel=muxA, pos=self.nInstr)
        #self.nInstr+=1
        # Return read and write srf indexes
        word = LSU_IMEM_WORD(mem_op=mem_op, vwr_sel_shuf_op=vwr_sel_shuf_op, rf_wsel=rf_wsel, rf_we=rf_we, alu_op=alu_op, muxb_sel=muxB, muxa_sel=muxA)
        return srf_read_index, srf_str_index, word
