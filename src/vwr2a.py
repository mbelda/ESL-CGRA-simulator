
# Imports
from src import *
from .params import *
from .spm import SPM


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

    def setSPMLine(self, nline, vector):
        self.spm.setLine(nline, vector)
    
    def kernel_config(self, column_usage, kernel_nInstr, imem_add_start, srf_spm_addres, kernel_number):
        self.kmem.addKernel(num_instructions=kernel_nInstr, imem_add_start=imem_add_start, column_usage=column_usage, srf_spm_addres=srf_spm_addres, nKernel=kernel_number)
        
    