"""rc.py: Data structures and objects emulating a Reconfigurable Cell of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
from enum import Enum
from ctypes import c_int32
import re

from .srf import SRF_N_REGS

# Local data register (DREG) sizes of specialized slots
RC_NUM_DREG = 2

# Configuration register (CREG) / instruction memory sizes of specialized slots
RC_NUM_CREG = 64

# Widths of instructions of each specialized slot in bits
RC_IMEM_WIDTH = 18

# RC IMEM word decoding
class RC_ALU_OPS(int, Enum):
    '''RC ALU operation codes'''
    NOP = 0
    SADD = 1
    SSUB = 2
    SMUL = 3
    SDIV = 4
    SLL = 5
    SRL = 6
    SRA = 7
    LAND = 8
    LOR = 9
    LXOR = 10
    INB_SF_INA = 11
    INB_ZF_INA = 12
    FXP_MUL = 13
    FXP_DIV = 14

class RC_MUX_SEL(int, Enum):
    '''Input A and B to RC ALU'''
    VWR_A = 0
    VWR_B = 1
    VWR_C = 2
    SRF = 3
    R0 = 4
    R1 = 5
    RCT = 6
    RCB = 7
    RCL = 8
    RCR = 9
    ZERO = 10
    ONE = 11
    MAX_INT = 12
    MIN_INT = 13

class RC_MUXF_SEL(int, Enum):
    '''Select the ALU origin of the data on which to compute flags for SF and ZF operations'''
    OWN = 0
    RCT = 1
    RCB = 2
    RCL = 3
    RCR = 4

class RC_DEST_REGS(int, Enum):
    '''Available registers to store ALU result'''
    R0 = 0
    R1 = 1
    SRF = 2
    VWR = 3

# RECONFIGURABLE CELL (RC) #

class RC_IMEM:
    '''Instruction memory of the Reconfigurable Cell'''
    def __init__(self):
        self.IMEM = np.zeros(RC_NUM_CREG,dtype="S{0}".format(RC_IMEM_WIDTH))
        # Initialize kernel memory with default word
        default_word = RC_IMEM_WORD()
        for i, instruction in enumerate(self.IMEM):
            self.IMEM[i] = default_word.get_word()
    
    def set_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary imem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=RC_IMEM_WIDTH)
    
    def set_params(self, rf_wsel=0, rf_we=0, muxf_sel=RC_MUXF_SEL.OWN, alu_op=RC_ALU_OPS.NOP, op_mode=0, muxb_sel=RC_MUX_SEL.VWR_A, muxa_sel=RC_MUX_SEL.VWR_A, pos=0):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See RC_IMEM_WORD initializer for implementation details.
        '''
        imem_word = RC_IMEM_WORD(rf_wsel=rf_wsel, rf_we=rf_we, muxf_sel=muxf_sel, alu_op=alu_op, op_mode=op_mode, muxb_sel=muxb_sel, muxa_sel=muxa_sel)
        self.IMEM[pos] = imem_word.get_word()
    
    def get_instruction_info(self, pos):
        '''Print the human-readable instructions of the instruction at position pos in the instruction memory'''
        imem_word = RC_IMEM_WORD()
        imem_word.set_word(self.IMEM[pos])
        rf_wsel, rf_we, muxf_sel, alu_op, op_mode, muxb_sel, muxa_sel = imem_word.decode_word()
        
        
        if op_mode==0:
            precision = "32-bit"
        else:
            precision = "16-bit"
        
        
        for op in RC_ALU_OPS:
            if op.value == alu_op:
                alu_opcode = op.name
        for sel in RC_MUX_SEL:
            if sel.value == muxa_sel:
                muxa_res = sel.name
        for sel in RC_MUX_SEL:
            if sel.value == muxb_sel:
                muxb_res = sel.name
        for sel in RC_MUXF_SEL:
            if sel.value == muxf_sel:
                muxf_res = sel.name
                
        if alu_opcode == RC_ALU_OPS.NOP.name:
            print("No ALU operation")
        elif (alu_opcode == RC_ALU_OPS.INB_SF_INA.name):
            print("Output {0} if sign flag of {1} == 1, else output {2}".format(muxa_res, muxf_res, muxb_res))
        elif (alu_opcode == RC_ALU_OPS.INB_ZF_INA.name):
            print("Output {0} if zero flag of {1} == 1, else output {2}".format(muxa_res, muxf_res, muxb_res))
        else:
            print("Performing ALU operation {0} between operands {1} and {2}".format(alu_opcode, muxa_res, muxb_res))
            print("ALU is performing operations with {0} precision".format(precision))
        
        if rf_we == 1:
            print("Writing ALU result to RC register {0}".format(rf_wsel))
        else:
            print("No RC registers are being written")
        
        
        
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the RC config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class RC_IMEM_WORD:
    def __init__(self, hex_word=None, rf_wsel=0, rf_we=0, muxf_sel=RC_MUXF_SEL.OWN, alu_op=RC_ALU_OPS.NOP, op_mode=0, muxb_sel=RC_MUX_SEL.VWR_A, muxa_sel=RC_MUX_SEL.VWR_A):
        '''Generate a binary rc instruction word from its configuration paramerers:
        
           -   rf_wsel: Select one of eight RC registers to write to
           -   rf_we: Enable writing to aforementioned register
           -   muxf_sel: Select a source for the “flag” parameter that is used to compute the zero and sign flags for some ALU operations
           -   alu_op: Perform one of the ALU operations listed in the RC_ALU_OPS enum
           -   op_mode: Constant 0 for now
           -   muxb_sel: Select input B to ALU (see RC_MUX_SEL enum for options)
           -   muxa_sel: Select input A to ALU (see RC_MUX_SEL enum for options)
        
        '''
        if hex_word == None:
            self.rf_wsel = np.binary_repr(rf_wsel, width=1)
            self.rf_we = np.binary_repr(rf_we,width=1)
            self.muxf_sel = np.binary_repr(muxf_sel,width=3)
            self.alu_op = np.binary_repr(alu_op,4)
            self.op_mode = np.binary_repr(op_mode,width=1)
            self.muxb_sel = np.binary_repr(muxb_sel,4)
            self.muxa_sel = np.binary_repr(muxa_sel,4)
            self.word = "".join((self.muxa_sel,self.muxb_sel,self.op_mode,self.alu_op,self.muxf_sel,self.rf_we,self.rf_wsel))
        else:
            decimal_int = int(hex_word, 16)
            binary_string = bin(decimal_int)[2:]  # Removing the '0b' prefix
            self.rf_wsel = binary_string[17:18] # 1 bit
            self.rf_we = binary_string[16:17] # 1 bit
            self.muxf_sel = binary_string[13:16] # 3 bits
            self.alu_op = binary_string[9:13] # 4 bits
            self.op_mode = binary_string[8:9] # 1 bit
            self.muxb_sel = binary_string[4:8] # 4 bits
            self.muxa_sel = binary_string[:4] # 4 bits
            self.word = binary_string

    def get_word(self):
        return self.word
    
    def get_word_in_hex(self):
        '''Get the hexadecimal representation of the word at index pos in the RC config IMEM'''
        return(hex(int(self.word, 2)))
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.rf_wsel = word[17:]
        self.rf_we = word[16:17]
        self.muxf_sel = word[13:16]
        self.alu_op = word[9:13]
        self.op_mode = word[8:9]
        self.muxb_sel = word[4:8]
        self.muxa_sel = word[0:4]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''
        rf_wsel = int(self.rf_wsel,2)
        rf_we = int(self.rf_we,2)
        muxf_sel = int(self.muxf_sel,2)
        alu_op = int(self.alu_op,2)
        op_mode = int(self.op_mode,2)
        muxb_sel = int(self.muxb_sel,2)
        muxa_sel = int(self.muxa_sel,2)
        
        
        return rf_wsel, rf_we, muxf_sel, alu_op, op_mode, muxb_sel, muxa_sel
    
class RC:
    rc_arith_ops   = {  'SADD','SSUB','SMUL','SDIV','SLL','SRL','SRA','LAND','LOR','SADD.H','SSUB.H','SMUL.H','SDIV.H','SLL.H','SRL.H','SRA.H','LAND.H','LOR.H','MUL.FP','DIV.FP' }
    rc_flag_ops     = { 'SFGA','ZFGA' }
    rc_nop_ops      = { 'NOP' }

    def __init__(self):
        self.regs       = [0 for _ in range(RC_NUM_DREG)]
        self.imem       = RC_IMEM()
        self.nInstr     = 0
        self.default_word = RC_IMEM_WORD().get_word()
    
    def run(self, pc):
        print(self.__class__.__name__ + ": " + self.imem.get_word_in_hex(pc))
        pass

    # def sadd( val1, val2 ):
    #     return c_int32( val1 + val2 ).value

    # def ssub( val1, val2 ):
    #     return c_int32( val1 - val2 ).value

    # def sll( val1, val2 ):
    #     return c_int32(val1 << val2).value

    # def srl( val1, val2 ):
    #     interm_result = (c_int32(val1).value & MAX_32b)
    #     return c_int32(interm_result >> val2).value

    # def sra( val1, val2 ):
    #     return c_int32(val1 >> val2).value

    # def lor( val1, val2 ):
    #     return c_int32( val1 | val2).value

    # def land( val1, val2 ):
    #     return c_int32( val1 & val2).value

    # def lxor( val1, val2 ):
    #     return c_int32( val1 ^ val2).value
    
    # def smul(self):
    #     pass

    # def sdiv(self, imm):
    #     pass
    
    # def saddh( val1, val2 ):
    #     raise ValueError("Half precision add not supported.")

    # def ssubh( val1, val2 ):
    #     raise ValueError("Half precision sub not supported.")

    # def sllh( val1, val2 ):
    #     raise ValueError("Half precision sll not supported.")

    # def srlh( val1, val2 ):
    #     raise ValueError("Half precision srl not supported.")
    
    # def srah( val1, val2 ):
    #     raise ValueError("Half precision sra not supported.")

    # def lorh( val1, val2 ):
    #     raise ValueError("Half precision lor not supported.")

    # def landh( val1, val2 ):
    #     raise ValueError("Half precision land not supported.")

    # def lxorh( val1, val2 ):
    #     raise ValueError("Half precision lxor not supported.")

    # def smulh(self):
    #     raise ValueError("Half precision mul not supported.")

    # def sdivh(self, imm):
    #     raise ValueError("Half precision div not supported.")

    # def nop(self):
    #     pass # Intentional

    # def mul_fp(self):
    #     pass

    # def div_fp(self, imm):
    #     pass

    # def sfga(self):
    #     pass

    # def zfga(self, imm):
    #     pass

    def parseDestArith(self, rd, instr):
        # Define the regular expression pattern
        r_pattern = re.compile(r'^R(\d+)$')
        srf_pattern = re.compile(r'^SRF\((\d+)\)$')
        vwr_pattern = re.compile(r'^VWR_([A-Za-z])$')

        # Check if the input matches the 'R' pattern
        r_match = r_pattern.match(rd)
        if r_match:
            ret = None
            try:
                ret = RC_DEST_REGS[rd]
            except:
                raise ValueError("Instruction not valid for RC: " + instr + ". The accessed register must be betwwen 0 and " + str(RC_NUM_DREG -1) + ".")
            return ret, -1, -1

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rd)
        if srf_match:
            return RC_DEST_REGS["SRF"], int(srf_match.group(1)), -1
        
        # Check if the input matches the 'VWR' pattern
        vwr_match = vwr_pattern.match(rd)
        if vwr_match:
            if vwr_match.group(1) == 'A':
                return RC_DEST_REGS["VWR"], -1, 0
            if vwr_match.group(1) == 'B':
                return RC_DEST_REGS["VWR"], -1, 1
            if vwr_match.group(1) == 'C':
                return RC_DEST_REGS["VWR"], -1, 2

        return None, -1, -1, -1

    # Returns the value for muxA and the number of the srf accessed (-1 if it isn't accessed)
    def parseMuxArith(self, rs, instr):
        # Define the regular expression pattern
        r_pattern = re.compile(r'^R(\d+)$')
        srf_pattern = re.compile(r'^SRF\((\d+)\)$')
        vwr_pattern = re.compile(r'^VWR_([A-Za-z])$')
        zero_pattern = re.compile(r'^ZERO$')
        one_pattern = re.compile(r'^ONE$')
        maxInt_pattern = re.compile(r'^MAX_INT$')
        minInt_pattern = re.compile(r'^MIN_INT$')
        neigh_pattern = re.compile(r'^RC([A-Za-z])$')

        # Check if the input matches the 'R' pattern
        r_match = r_pattern.match(rs)
        if r_match:
            ret = None
            try:
                ret = RC_MUX_SEL[rs]
            except:
                raise ValueError("Instruction not valid for RC: " + instr + ". The accessed register must be between 0 and " + str(RC_NUM_DREG -1) + ".")
            return ret, -1

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rs)
        if srf_match:
            i = srf_match.group(1)
            return RC_MUX_SEL["SRF"], int(srf_match.group(1))
        
        # Check if the input matches the 'VWR' pattern
        vwr_match = vwr_pattern.match(rs)
        if vwr_match:
            try:
                ret = RC_MUX_SEL[rs]
            except:
                raise ValueError("Instruction not valid for RC: " + instr + ". The accessed VWR must be A, B or C.")
            return ret, -1
            
        # Check if the input matches the 'RCX' pattern
        neigh_match = neigh_pattern.match(rs)
        if neigh_match:
            ret = None
            try:
                ret = RC_MUX_SEL[rs]
            except:
                raise ValueError("Instruction not valid for RC: " + instr + ". The accessed register is not a valid neighbour (RCT, RCB, RCR, RCL).")
            return ret, -1
        
        # Check if the input matches the 'ZERO' pattern
        zero_match = zero_pattern.match(rs)
        if zero_match:
            return RC_MUX_SEL[rs], -1

        # Check if the input matches the 'ONE' pattern
        one_match = one_pattern.match(rs)
        if one_match:
            return RC_MUX_SEL[rs], -1
        
        # Check if the input matches the 'MAX_INT' pattern
        maxInt_match = maxInt_pattern.match(rs)
        if maxInt_match:
            return RC_MUX_SEL[rs], -1
        
        # Check if the input matches the 'MIN_INT' pattern
        minInt_match = minInt_pattern.match(rs)
        if minInt_match:
            return RC_MUX_SEL[rs], -1

        return None, -1
    
    def parseFlag(self, flag, instr):
        ret = None
        try:
            ret = RC_MUXF_SEL[flag]
        except:
            raise ValueError("Instruction not valid for RC: " + instr + ". The accessed ALU flags parameters is not valid (OWN, RCT, RCB, RCR, RCL).")

        return ret

    def asmToHex(self, instr):
        space_instr = instr.replace(",", " ")
        split_instr = [word for word in space_instr.split(" ") if word]
        try:
            op      = split_instr[0]
        except:
            op      = split_instr

        if op in self.rc_arith_ops:
            
            if '.H' in op:
                raise ValueError("Half precision not supported yet.")
            elif '.FP' in op:
                if op == "DIV.FP":
                    raise ValueError("Float point division not supported yet.")
                if op == "MUL.FP":
                    alu_op = RC_ALU_OPS["FXP_MUL"]
            else:
                alu_op = RC_ALU_OPS[op]
            # Expect 3 operands: rd/srf, rs/srf/zero/one, rt/srf/zero/imm
            try:
                rd = split_instr[1]
                rs = split_instr[2]
                rt = split_instr[3]
            except:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected 3 operands.")
            dest, srf_str_index, vwr_str = self.parseDestArith(rd, instr)
            muxA, srf_read_index = self.parseMuxArith(rs, instr)
            muxB, srf_muxB_index = self.parseMuxArith(rt, instr)

            if srf_read_index >= SRF_N_REGS or srf_muxB_index >= SRF_N_REGS or srf_str_index >= SRF_N_REGS:
                raise ValueError("Instruction not valid for RC: " + instr + ". The accessed SRF must be between 0 and " + str(SRF_N_REGS -1) + ".")

            if dest == None:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected another format for first operand (dest).")
            
            if muxA == None:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected another format for the second operand (muxA).")

            if muxB == None:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected another format for the third operand (muxB).")
            
            if srf_muxB_index != -1:
                if srf_read_index != -1 and srf_muxB_index != srf_read_index:
                    raise ValueError("Instruction not valid for RC: " + instr + ". Expected only reads/writes to the same reg of the SRF.") 
                srf_read_index = srf_muxB_index

            if srf_str_index != -1 and srf_read_index != -1 and srf_str_index != srf_read_index:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected only reads/writes to the same reg of the SRF.")

            if srf_str_index == -1 and vwr_str == -1: # Writting on a  local reg
                rf_we = 1
                rf_wsel = dest
            else:
                rf_wsel = 0
                rf_we = 0

            op_mode = 0
            muxf_sel=RC_MUXF_SEL.OWN

            # Add hexadecimal instruction
            #self.imem.set_params(rf_wsel=rf_wsel, rf_we=rf_we, muxf_sel=muxf_sel, alu_op=alu_op, op_mode=op_mode, muxb_sel=muxB, muxa_sel=muxA, pos=self.nInstr)
            #self.nInstr+=1
            # Return read and write srf indexes and the flag to write on a vwr
            word = RC_IMEM_WORD(rf_wsel=rf_wsel, rf_we=rf_we, muxf_sel=muxf_sel, alu_op=alu_op, op_mode=op_mode, muxb_sel=muxB, muxa_sel=muxA)
            return srf_read_index, srf_str_index, vwr_str, word
        
        if op in self.rc_nop_ops:
            alu_op = RC_ALU_OPS[op]
            # Expect 0 operands
            if len(split_instr) > 1:
                raise ValueError("Instruction not valid for RC: " + instr + ". Nop does not expect operands.")
            
            #self.imem.set_params(alu_op=alu_op, pos=self.nInstr)
            #self.nInstr+=1
            # Return read and write srf indexes
            word = RC_IMEM_WORD(alu_op=alu_op)
            return -1, -1, -1, word
        
        if op in self.rc_flag_ops:
            if op == "SFGA":
                alu_op = RC_ALU_OPS["INB_SF_INA"]
            if op == "ZFGA":
                alu_op = RC_ALU_OPS["INB_ZF_INA"]

            # Expect 2 operands
            try:
                rd = split_instr[1]
                flag = split_instr[2]
            except:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected 2 operands.")
            
            dest, srf_str_index, vwr_str = self.parseDestArith(rd, instr)
            muxf_sel = self.parseFlag(flag, instr)

            if dest == None:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected another format for first operand (dest).")
            
            if muxf_sel == None:
                raise ValueError("Instruction not valid for RC: " + instr + ". Expected another format for second operand (flag).")

            if srf_str_index == -1 and vwr_str == -1: # Writting on a  local reg
                rf_we = 1
                rf_wsel = dest
            else:
                rf_wsel = 0
                rf_we = 0

            # Add hexadecimal instruction
            #self.imem.set_params(rf_wsel=rf_wsel, rf_we=rf_we, muxf_sel=muxf_sel, alu_op=alu_op, pos=self.nInstr)
            #self.nInstr+=1
            # Return read and write srf indexes and the flag to write on a vwr
            word = RC_IMEM_WORD(rf_wsel=rf_wsel, rf_we=rf_we, muxf_sel=muxf_sel, alu_op=alu_op)
            return -1, srf_str_index, vwr_str, word
        

        raise ValueError("Instruction not valid for RC: " + instr + ". Operation not recognised.")
