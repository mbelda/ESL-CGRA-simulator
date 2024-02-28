
# Imports
from src import *
from .params import *
from .spm import SPM
from .srf import SRF


class CGRA:
    def __init__(self):
        self.lcus = [LCU() for _ in range(CGRA_COLS)]
        self.lsus = [LSU() for _ in range(CGRA_COLS)]
        self.rcs = [[] for _ in range(CGRA_COLS)]
        for col in range(CGRA_COLS):
            for _ in range(CGRA_ROWS):
                self.rcs[col].append(RC())
        self.mxcus = [MXCU() for _ in range(CGRA_COLS)]
        self.spm = SPM()
        self.kmem = KMEM()
        self.imem = IMEM()
        self.srfs = [SRF() for _ in range(CGRA_COLS)]

    def setSPMLine(self, nline, vector):
        self.spm.setLine(nline, vector)
    
    def loadSPMData(self, data):
        nline = 0
        for vector in data:
            self.spm.setLine(nline, vector)
            nline+=1
    
    def kernel_config(self, col_one_hot, kernel_nInstr, imem_add_start, srf_spm_addres, kernel_number):
        self.kmem.addKernel(num_instructions=kernel_nInstr, imem_add_start=imem_add_start, col_one_hot=col_one_hot, srf_spm_addres=srf_spm_addres, nKernel=kernel_number)
        
    