from .params import *

class VWR():
    def __init__(self):
        self.values = [0 for _ in range(N_ELEMS_PER_VWR)]
    
    def getIdx(self, idx):
        return self.values[idx]