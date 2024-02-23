# Scratchpad memory configuration
SPM_NWORDS = 128
SPM_NLINES = 64

class SPM:
    def __init__(self):
        self.lines = [[0 for _ in range(SPM_NWORDS)] for _ in range(SPM_NLINES)]
    
    def setLine(self, nline, vec):
        assert(nline >= 0 & nline < SPM_NLINES), "SPM: Number of SPM line out of bounds. It should be >= 0 and < " + str(SPM_NLINES) + "."
        assert(len(vec) == SPM_NWORDS), "SPM: Vector should have " + str(SPM_NWORDS) + " elements."
        self.lines[nline] = vec
    
    def getLine(self, nline):
        assert(nline >= 0 & nline < SPM_NLINES), "SPM: Number of SPM line out of bounds. It should be >= 0 and < " + str(SPM_NLINES) + "."
        return self.lines[nline]