"""lcu.py: Data structures and objects emulating the Loop Control Unit of the VWR2A architecture"""
__author__      = "Lara Orlandic"
__email__       = "lara.orlandic@epfl.ch"

import numpy as np
from enum import Enum
from ctypes import c_int32
import re


# Local data register (DREG) sizes of specialized slots
LCU_NUM_DREG = 4

# Configuration register (CREG) / instruction memory sizes of specialized slots
LCU_NUM_CREG = 64

# Widths of instructions of each specialized slot in bits
LCU_IMEM_WIDTH = 20

# Number of scalar registers shared on the column
SRF_N_REGS = 8

# LCU IMEM word decoding
class LCU_ALU_OPS(int, Enum):
    '''LCU ALU operation codes'''
    NOP = 0
    SADD = 1
    SSUB = 2
    SLL = 3
    SRL = 4
    SRA = 5
    LAND = 6
    LOR = 7
    LXOR = 8
    BEQ = 9
    BNE = 10
    BGEPD = 11
    BLT = 12
    JUMP = 13
    EXIT = 14

class LCU_DEST_REGS(int, Enum):
    '''Available ALU registers to store ALU result'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    SRF = 4

class LCU_MUXA_SEL(int, Enum):
    '''Input A to LCU ALU'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    SRF = 4
    LAST = 5
    ZERO = 6
    IMM = 7

class LCU_MUXB_SEL(int, Enum):
    '''Input B to LCU ALU'''
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    SRF = 4
    LAST = 5
    ZERO = 6
    ONE = 7
    
# LOOP CONTROL UNIT (LCU) #

class LCU_IMEM:
    '''Instruction memory of the Loop Control Unit'''
    def __init__(self):
        self.IMEM = np.zeros(LCU_NUM_CREG,dtype="S{0}".format(LCU_IMEM_WIDTH))
        # Initialize kernel memory with default instruction
        default_word = LCU_IMEM_WORD()
        for i in range(LCU_NUM_CREG):
            self.IMEM[i] = default_word.get_word()
    
    def set_word(self, kmem_word, pos):
        '''Set the IMEM index at integer pos to the binary imem word'''
        self.IMEM[pos] = np.binary_repr(kmem_word,width=LCU_IMEM_WIDTH)
    
    def set_params(self, imm=0, rf_wsel=0, rf_we=0, alu_op=LCU_ALU_OPS.NOP, br_mode=0, muxb_sel=LCU_MUXB_SEL.R0, muxa_sel=LCU_MUXA_SEL.R0, pos=0):
        '''Set the IMEM index at integer pos to the configuration parameters.
        See LCU_IMEM_WORD initializer for implementation details.
        '''
        imem_word = LCU_IMEM_WORD(imm, rf_wsel, rf_we, alu_op, br_mode, muxb_sel, muxa_sel)
        self.IMEM[pos] = imem_word.get_word()
    
    def get_instruction_info(self, pos):
        '''Print the human-readable instructions of the instruction at position pos in the instruction memory'''
        imem_word = LCU_IMEM_WORD()
        imem_word.set_word(self.IMEM[pos])
        imm, rf_wsel, rf_we, alu_op, br_mode, muxb_sel, muxa_sel = imem_word.decode_word()
        
        print("Immediate value: {0}".format(imm))
        
        if br_mode == 1:
            print ("LCU is in RC data control mode")
        else: 
            print ("LCU is in loop control mode")
            
        for op in LCU_ALU_OPS:
            if op.value == alu_op:
                alu_opcode = op.name
        for sel in LCU_MUXA_SEL:
            if sel.value == muxa_sel:
                muxa_res = sel.name
        for sel in LCU_MUXB_SEL:
            if sel.value == muxb_sel:
                muxb_res = sel.name
        if alu_op == 0: #NOP
            print("No LCU ALU Operation is performed")
        elif alu_op == 9: #BEQ
            print("If {0} and {1} are equal, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 10: #BNE
            print("If {0} and {1} are NOT equal, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 11: #BGEPD
            print("If {0}-1 is greater than or equal to {1}, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 12: #BLT
            print("If {0} is less than {1}, branch to the immediate value {2}".format(muxa_res, muxb_res, imm))
        elif alu_op == 13: #JUMP
            print("Jump to address {0} + {1}".format(muxa_res, muxb_res))
        elif alu_op == 14: #EXIT
            print("Exiting out of kernel")
        else:
            print("Performing ALU operation {0} between operands {1} and {2}".format(alu_opcode, muxa_res, muxb_res))
        
        if rf_we == 1:
            print("Writing ALU result to LCU register {0}".format(rf_wsel))
        else:
            print("No LCU registers are being written")

            
    def get_word_in_hex(self, pos):
        '''Get the hexadecimal representation of the word at index pos in the LCU config IMEM'''
        return(hex(int(self.IMEM[pos],2)))
        
    
        
class LCU_IMEM_WORD:
    def __init__(self, imm=0, rf_wsel=0, rf_we=0, alu_op=LCU_ALU_OPS.NOP, br_mode=0, muxb_sel=LCU_MUXB_SEL.R0, muxa_sel=LCU_MUXA_SEL.R0):
        '''Generate a binary lcu instruction word from its configuration paramerers:
        
           -   imm: Immediate value to use for ALU operations or address to branch to
           -   rf_wsel: Select one of four LCU registers to write to
           -   rf_we: Enable writing to aforementioned register
           -   alu_op: Perform one of the ALU operations listed in the LCU_ALU_OPS enum
           -   br_mode: Control program counter (0) or RC datapath (1)
           -   muxb_sel: Select input B to ALU (see LCU_MUXB_SEL enum for options)
           -   muxa_sel: Select input A to ALU (see LCU_MUXA_SEL enum for options)
        
        '''
        self.imm = np.binary_repr(imm, width=6)
        self.rf_wsel = np.binary_repr(rf_wsel, width=2)
        self.rf_we = np.binary_repr(rf_we,width=1)
        self.alu_op = np.binary_repr(alu_op,4)
        self.br_mode = np.binary_repr(br_mode,1)
        self.muxb_sel = np.binary_repr(muxb_sel,3)
        self.muxa_sel = np.binary_repr(muxa_sel,3)
        self.word = "".join((self.muxa_sel,self.muxb_sel,self.br_mode,self.alu_op,self.rf_we,self.rf_wsel,self.imm))
    
    def get_word(self):
        return self.word
    
    def set_word(self, word):
        '''Set the binary configuration word of the kernel memory'''
        self.word = word
        self.imm = word[14:]
        self.rf_wsel = word[12:14]
        self.rf_we = word[11:12]
        self.alu_op = word[7:11]
        self.br_mode = word[6:7]
        self.muxb_sel = word[3:6]
        self.muxa_sel = word[0:3]
        
    
    def decode_word(self):
        '''Get the configuration word parameters from the binary word'''
        imm = int(self.imm,2)
        rf_wsel = int(self.rf_wsel,2)
        rf_we = int(self.rf_we,2)
        alu_op = int(self.alu_op,2)
        br_mode = int(self.br_mode,2)
        muxb_sel = int(self.muxb_sel,2)
        muxa_sel = int(self.muxa_sel,2)
        
        return imm, rf_wsel, rf_we, alu_op, br_mode, muxb_sel, muxa_sel


class LCU:
    def __init__(self):
        self.regs       = {'R0':0, 'R1':0, 'R2':0, 'R3':0 } # Parametrize
        self.imem       = LCU_IMEM()
        self.nInstr     = 0
        self.default_word = LCU_IMEM_WORD().get_word()
    
    def sadd( val1, val2 ):
        return c_int32( val1 + val2 ).value

    def ssub( val1, val2 ):
        return c_int32( val1 - val2 ).value

    def sll( val1, val2 ):
        return c_int32(val1 << val2).value

    def srl( val1, val2 ):
        interm_result = (c_int32(val1).value & MAX_32b)
        return c_int32(interm_result >> val2).value

    def sra( val1, val2 ):
        return c_int32(val1 >> val2).value

    def lor( val1, val2 ):
        return c_int32( val1 | val2).value

    def land( val1, val2 ):
        return c_int32( val1 & val2).value

    def lxor( val1, val2 ):
        return c_int32( val1 ^ val2).value

    def beq( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 == val2 else self.flags['branch']

    def bne( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 != val2 else self.flags['branch']

    def bgepd( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 >= val2 else self.flags['branch']

    def blt( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 < val2 else self.flags['branch']

    def beqr( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 == val2 else self.flags['branch']

    def bner( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 != val2 else self.flags['branch']

    def bger( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 >= val2 else self.flags['branch']

    def bltr( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 < val2 else self.flags['branch']

    def nop(self):
        pass # Intentional

    def exit(self):
        pass

    def jump(self, imm):
        pass

    def parseDestArith(self, rd):
        # Define the regular expression pattern
        r_pattern = re.compile(r'^R(\d+)$')
        srf_pattern = re.compile(r'^SRF\((\d+)\)$')

        # Check if the input matches the 'R' pattern
        r_match = r_pattern.match(rd)
        if r_match:
            return LCU_DEST_REGS[rd], -1

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rd)
        if srf_match:
            return LCU_DEST_REGS["SRF"], srf_match.group(1)

        return None, -1

    # Returns the value for muxA and the number of the srf accessed (-1 if it isn't accessed)
    def parseMuxAArith(self, rs):
        # Define the regular expression pattern
        r_pattern = re.compile(r'^R(\d+)$')
        srf_pattern = re.compile(r'^SRF\((\d+)\)$')
        zero_pattern = re.compile(r'^ZERO$')
        last_pattern = re.compile(r'^LAST$')

        # Check if the input matches the 'R' pattern
        r_match = r_pattern.match(rs)
        if r_match:
            return LCU_MUXA_SEL[rs], -1

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rs)
        if srf_match:
            i = srf_match.group(1)
            return LCU_MUXA_SEL["SRF"], srf_match.group(1)
        
        # Check if the input matches the 'ZERO' pattern
        zero_match = zero_pattern.match(rs)
        if zero_match:
            return LCU_MUXA_SEL[rs], -1

        # Check if the input matches the 'LAST' pattern
        last_match = last_pattern.match(rs)
        if last_match:
            return LCU_MUXA_SEL[rs], -1

        return None, -1

    def parseMuxBArith(self, rs):
        # Define the regular expression pattern
        r_pattern = re.compile(r'^R(\d+)$')
        srf_pattern = re.compile(r'^SRF\((\d+)\)$')
        zero_pattern = re.compile(r'^ZERO$')
        last_pattern = re.compile(r'^LAST$')
        one_pattern = re.compile(r'^ONE$')

        # Check if the input matches the 'R' pattern
        r_match = r_pattern.match(rs)
        if r_match:
            return LCU_MUXB_SEL[rs], -1

        # Check if the input matches the 'SRF' pattern
        srf_match = srf_pattern.match(rs)
        if srf_match:
            return LCU_MUXB_SEL["SRF"], srf_match.group(1)
        
        # Check if the input matches the 'ZERO' pattern
        zero_match = zero_pattern.match(rs)
        if zero_match:
            return LCU_MUXB_SEL[rs], -1

        # Check if the input matches the 'LAST' pattern
        last_match = last_pattern.match(rs)
        if last_match:
            return LCU_MUXB_SEL[rs], -1
        
        # Check if the input matches the 'ONE' pattern
        one_match = one_pattern.match(rs)
        if one_match:
            return LCU_MUXB_SEL[rs], -1

        return None, -1

    def asmToHex(self, instr):
        space_instr = instr.replace(",", " ")
        split_instr = [word for word in space_instr.split(" ") if word]
        try:
            op      = split_instr[0]
        except:
            op      = split_instr

        if op in self.lcu_arith_ops:
            alu_op = LCU_ALU_OPS[op]
            # Expect 3 operands: rd/srf, rs/srf/zero/one, rt/srf/zero/imm
            try:
                rd = split_instr[1]
                rs = split_instr[2]
                rt = split_instr[3]
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected 3 operands.")
            dest, srf_str_index = self.parseDestArith(rd)
            muxB, srf_muxB_index = self.parseMuxBArith(rs) # Change order so that always the ONE value can be written in the first operand in the assembly
            muxA, srf_read_index = self.parseMuxAArith(rt)

            if srf_read_index > SRF_N_REGS or srf_muxB_index > SRF_N_REGS or srf_str_index > SRF_N_REGS:
                raise ValueError("Instruction not valid for LCU: " + instr + ". The accessed SRF must be betwwen 0 and " + str(SRF_N_REGS -1) + ".")

            if dest == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for first operand (dest).")
            
            if muxB == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for the second operand (muxB).")

            if muxA == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for the third operand (muxA).")
            
            if srf_muxB_index != -1:
                if srf_read_index != -1 and srf_muxB_index != srf_read_index:
                    raise ValueError("Instruction not valid for LCU: " + instr + ". Expected only reads/writes to the same reg of the SRF.") 
                srf_read_index = srf_muxB_index

            if srf_str_index != -1 and srf_read_index != -1 and srf_str_index != srf_read_index:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected only reads/writes to the same reg of the SRF.")

            br_mode = 0
            if srf_str_index == -1: # Writting on a  local reg
                rf_we = 1
                rf_wsel = dest
            else:
                rf_wsel = 0
                rf_we = 0
            imm = 0

            # Add hexadecimal instruction
            self.imem.set_params(imm=imm, rf_wsel=rf_wsel, rf_we=rf_we, alu_op=alu_op, br_mode=br_mode, muxb_sel=muxB, muxa_sel=muxA, pos=self.nInstr)
            self.nInstr+=1
            # Return read and write srf indexes
            return srf_read_index, srf_str_index
        
        if op in self.lcu_arith_i_ops:
            alu_op = LCU_ALU_OPS[op[:-1]]
            # Expect 3 operands: rd/srf, rs/srf/zero/one, imm
            try:
                rd = split_instr[1]
                rs = split_instr[2]
                rt = split_instr[3]
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected 3 operands.")
            dest, srf_str_index = self.parseDestArith(rd)
            muxA = LCU_MUXA_SEL["IMM"]
            muxB, srf_read_index = self.parseMuxBArith(rs)

            if srf_read_index > SRF_N_REGS or srf_str_index > SRF_N_REGS:
                raise ValueError("Instruction not valid for LCU: " + instr + ". The accessed SRF must be betwwen 0 and " + str(SRF_N_REGS -1) + ".")

            if dest == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for first operand (dest).")
            
            if muxB == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for the second operand (muxB).")
            
            try:
                imm = int(rt) 
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected an inmediate as third operand.")

            if srf_str_index != -1 and srf_read_index != -1 and srf_str_index != srf_read_index:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected only reads/writes to the same reg of the SRF.")

            if srf_str_index == -1: # Writting on a  local reg
                rf_we = 1
                rf_wsel = dest
            else:
                rf_wsel = 0
                rf_we = 0

            # Add hexadecimal instruction
            self.imem.set_params(imm=imm, rf_wsel=rf_wsel, rf_we=rf_we, alu_op=alu_op, muxb_sel=muxB, muxa_sel=muxA, pos=self.nInstr)
            self.nInstr+=1
            # Return read and write srf indexes
            return srf_read_index, srf_str_index

        if op in self.lcu_rcmode_ops:
            alu_op = LCU_ALU_OPS[op[:-1]]
            # Expect 1 operand: imm
            try:
                imm_str = split_instr[1]
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected 1 operand.")
            try:
                imm = int(imm_str) 
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected an inmediate as an operand.")
            
            br_mode = 1
            # Add hexadecimal instruction
            self.imem.set_params(imm=imm, alu_op=alu_op, br_mode=br_mode, pos=self.nInstr)
            self.nInstr+=1
            # Return read and write srf indexes
            return -1, -1

        if op in self.lcu_branch_ops:
            alu_op = LCU_ALU_OPS[op]
            # Expect 3 operands: rs/srf/zero/one, rs/srf/zero/imm, imm
            try:
                rs = split_instr[1]
                rt = split_instr[2]
                imm_str = split_instr[3]
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected 3 operands.")
            muxA, srf_muxA_index = self.parseMuxAArith(rt)
            muxB, srf_muxB_index = self.parseMuxBArith(rs)

            if srf_muxB_index > SRF_N_REGS or srf_muxA_index > SRF_N_REGS:
                raise ValueError("Instruction not valid for LCU: " + instr + ". The accessed SRF must be betwwen 0 and " + str(SRF_N_REGS -1) + ".")

            srf_str_index = -1
            if op == "BGEPD":
                srf_str_index = srf_muxB_index

            if muxB == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for the first operand (muxB).")
            
            if muxA == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for the second operand (muxA).")
            
            try:
                imm = int(imm_str) 
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected an inmediate as third operand.")

            if srf_muxA_index != -1 and srf_muxA_index != srf_read_index:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected only reads/writes to the same reg of the SRF.")
            srf_read_index = srf_muxA_index

            br_mode = 0
            # Add hexadecimal instruction
            self.imem.set_params(imm=imm, alu_op=alu_op, br_mode=br_mode, muxb_sel=muxB, muxa_sel=muxA, pos=self.nInstr)
            self.nInstr+=1
            # Return read and write srf indexes
            return srf_read_index, srf_str_index

        if op in self.lcu_jump_ops:
            alu_op = LCU_ALU_OPS[op]
            # Expect 2 operands: rs/srf/zero/one, rs/srf/zero/imm
            try:
                rs = split_instr[1]
                rt = split_instr[2]
            except:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected 2 operands.")
            muxB, srf_muxB_index = self.parseMuxBArith(rs) # Change order so that always the ONE value can be written in the first operand in the assembly
            muxA, srf_read_index = self.parseMuxAArith(rt)
            imm = 0

            if srf_muxB_index > SRF_N_REGS or srf_read_index > SRF_N_REGS:
                raise ValueError("Instruction not valid for LCU: " + instr + ". The accessed SRF must be betwwen 0 and " + str(SRF_N_REGS -1) + ".")

            if muxB == None:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for the first operand (muxB).")
            
            if muxA == None:
                try:
                    imm = int(rt)
                except:
                    raise ValueError("Instruction not valid for LCU: " + instr + ". Expected another format for the second operand (muxA).")
                muxA = LCU_MUXA_SEL["IMM"]
            
            if srf_muxB_index != -1:
                if srf_read_index != -1 and srf_muxB_index != srf_read_index:
                    raise ValueError("Instruction not valid for LCU: " + instr + ". Expected only reads/writes to the same reg of the SRF.") 
                srf_read_index = srf_muxB_index

            # Add hexadecimal instruction
            self.imem.set_params(imm=imm, alu_op=alu_op, muxb_sel=muxB, muxa_sel=muxA, pos=self.nInstr)
            self.nInstr+=1
            # Return read and write srf indexes
            return srf_read_index, -1

        if op in self.lcu_nop_ops:
            alu_op = LCU_ALU_OPS[op]
            # Expect 0 operands
            if len(split_instr) > 1:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Nop does not expect operands.")
            self.imem.set_params(alu_op=alu_op, pos=self.nInstr)
            self.nInstr+=1
            # Return read and write srf indexes
            return -1, -1
        
        if op in self.lcu_exit_ops:
            alu_op = LCU_ALU_OPS[op]
            # Expect 0 operands
            if len(split_instr) > 1:
                raise ValueError("Instruction not valid for LCU: " + instr + ". Exit does not expect operands.")
            self.imem.set_params(alu_op=alu_op, pos=self.nInstr)
            self.nInstr+=1
            # Return read and write srf indexes
            return -1, -1
        
        raise ValueError("Instruction not valid for LCU: " + instr + ". Operation not recognised.")

    lcu_arith_ops   = { 'SADD'      : sadd,
                        'SSUB'      : ssub,
                        'SLL'       : sll,
                        'SRL'       : srl,
                        'SRA'       : sra,
                        'LAND'      : land,
                        'LOR'       : lor,
                        'LXOR'      : lxor }
        
    lcu_arith_i_ops = { 'SADDI'      : sadd,
                        'SSUBI'      : ssub,
                        'SLLI'       : sll,
                        'SRLI'       : srl,
                        'SRAI'       : sra,
                        'LANDI'      : land,
                        'LORI'       : lor,
                        'LXORI'      : lxor }

    lcu_rcmode_ops  = { 'BEQR'       : beqr,
                        'BNER'       : bner,
                        'BLTR'       : bltr,
                        'BGER'       : bger }

    lcu_branch_ops  = { 'BEQ'       : beq,
                        'BNE'       : bne,
                        'BLT'       : blt,
                        'BGEPD'       : bgepd }

    lcu_nop_ops     = { 'NOP'       : nop }
    lcu_exit_ops    = { 'EXIT'      : exit }

    lcu_jump_ops    = { 'JUMP'      : jump }

    