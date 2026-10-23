import numpy as np
from ctypes import c_int32
import csv
import os.path
from dataclasses import dataclass

from kernels import *

N_ROWS = 4
N_COLS = 4
INSTR_SIZE = N_ROWS + 1
MAX_COL = N_COLS - 1
MAX_ROW = N_ROWS - 1

PRINT_OUTS  = 1

MAX_32b = 0xFFFFFFFF


srcs    = ['ZERO', 'SELF', 'RCL', 'RCR', 'RCT', 'RCB', 'R0', 'R1', 'R2', 'R3', 'IMM']
dsts    = ['SELF', 'RCL', 'RCR', 'RCT', 'RCB', 'R0', 'R1', 'R2', 'R3']
regs    = dsts[-4:]


@dataclass(frozen=True)
class ArchitectureProfile:
    """Geometry for the optional, LCU-equipped OpenEdge experiments.

    ``compute_cols`` counts only conventional RC columns.  On an LCU profile,
    one additional physical column per row is reserved for the LCU and is not
    part of the arithmetic neighbour network.
    """
    name: str
    rows: int
    compute_cols: int
    lcu_per_row: bool = False

    @property
    def physical_cols(self):
        return self.compute_cols + int(self.lcu_per_row)


ARCHITECTURES = {
    "OE_4x4": ArchitectureProfile("OE_4x4", 4, 4, False),
    "OE_2x5_LCU": ArchitectureProfile("OE_2x5_LCU", 2, 4, True),
    "OE_5x5_LCU": ArchitectureProfile("OE_5x5_LCU", 5, 4, True),
    # Matched-compute controls for estimating the LCU's contribution apart
    # from changing the number of arithmetic RCs.
    "OE_2x4_REF": ArchitectureProfile("OE_2x4_REF", 2, 4, False),
    "OE_5x4_REF": ArchitectureProfile("OE_5x4_REF", 5, 4, False),
}


@dataclass
class HeterogeneousPC:
    """One lockstep PC: one instruction per compute RC and optional LCU ops.

    ``pe_ops`` is a rows-by-compute-columns matrix of instruction strings.
    ``lcu_ops`` contains one instruction string per row, or ``None``.  LCU
    instructions use the same ALU/branch spelling as PE instructions and
    four private registers (R0--R3) per LCU.
    """
    pe_ops: list
    lcu_ops: list = None


class LCU:
    """One-cycle LCU context with four registers and a PC-local ALU/branch.

    Arithmetic, compare and branch resolve within the issued PC, like the
    combinational LCU datapath/PC branch path in DISCO.  The simulator issues
    an LCU operation concurrently with that PC's PE instructions; the LCU
    still consumes the PC's cycle and never runs while the CGRA PC is stalled.
    """
    def __init__(self, row):
        self.row = row
        self.regs = {"R0": 0, "R1": 0, "R2": 0, "R3": 0}

    def fetch(self, token):
        if token in self.regs:
            return self.regs[token]
        if token == "ZERO":
            return 0
        try:
            return int(token, 0)
        except ValueError as exc:
            raise ValueError(f"LCU row {self.row}: invalid operand {token!r}") from exc

    def execute(self, text, pc):
        if text is None or not str(text).strip() or str(text).strip().upper() == "NOP":
            return None
        words = str(text).replace(",", " ").split()
        op = words[0].upper()
        if op in ("SADD", "SSUB"):
            if len(words) != 4 or words[1] not in self.regs:
                raise ValueError(f"LCU row {self.row}: malformed {text!r}")
            a, b = self.fetch(words[2]), self.fetch(words[3])
            value = a + b if op == "SADD" else a - b
            self.regs[words[1]] = c_int32(value).value
            return None
        if op in ("BEQ", "BNE", "BLT", "BGE"):
            if len(words) != 4:
                raise ValueError(f"LCU row {self.row}: malformed {text!r}")
            a, b, target = self.fetch(words[1]), self.fetch(words[2]), self.fetch(words[3])
            taken = {"BEQ": a == b, "BNE": a != b,
                     "BLT": a < b, "BGE": a >= b}[op]
            return target if taken else None
        if op == "JUMP":
            if len(words) != 3:
                raise ValueError(f"LCU row {self.row}: expected JUMP base offset")
            return self.fetch(words[1]) + self.fetch(words[2])
        raise ValueError(f"LCU row {self.row}: unsupported instruction {op!r}")


class HeterogeneousCGRA:
    """Lockstep OpenEdge-like compute mesh with one LCU context per row.

    The LCU is a separate control endpoint, not an arithmetic RC.  Its
    operations are co-issued with the corresponding compute PC and evaluated
    once per modeled PC cycle.  All row LCUs must agree on a branch target;
    divergence is an error rather than an implicit row-local PC fork.
    """
    def __init__(self, architecture, program, memory=None,
                 read_addrs=None, write_addrs=None):
        if isinstance(architecture, str):
            try:
                architecture = ARCHITECTURES[architecture]
            except KeyError as exc:
                raise ValueError(f"Unknown architecture {architecture!r}") from exc
        self.architecture = architecture
        self.rows = architecture.rows
        self.compute_cols = architecture.compute_cols
        self.instrs = list(program)
        self.cells = [[PE(self, r, c) for c in range(self.compute_cols)]
                      for r in range(self.rows)]
        self.lcus = ([LCU(r) for r in range(self.rows)]
                     if architecture.lcu_per_row else [])
        self.memory = memory if memory is not None else []
        self.memory_index = {}
        for index, row in enumerate(self.memory):
            self.memory_index.setdefault(row[0], index)
        self.load_addr = list(read_addrs or [0] * self.compute_cols)
        self.store_addr = list(write_addrs or [0] * self.compute_cols)
        if len(self.load_addr) != self.compute_cols or len(self.store_addr) != self.compute_cols:
            raise ValueError("Direct memory pointers must have one entry per compute column")
        self.instr2exec = 0
        self.aproxcycles = 0
        self.aproxMemCycles = 0
        self.aproxArithCycles = 0
        self.exit = False
        self.cycles = 0
        self.lcu_operations = 0
        self.lcu_branch_operations = 0
        self.lcu_taken_branches = 0
        self.lcu_issue_pcs = 0

    def get_neighbour_address(self, r, c, direction):
        # LCUs are not RC neighbours.  Wrap-around applies only to the
        # rows-by-four compute mesh, including top/bottom links.
        if direction == "RCL": c = (c - 1) % self.compute_cols
        elif direction == "RCR": c = (c + 1) % self.compute_cols
        elif direction == "RCT": r = (r - 1) % self.rows
        elif direction == "RCB": r = (r + 1) % self.rows
        else: raise ValueError(f"Unknown neighbour direction {direction!r}")
        return r, c

    def get_neighbour_out(self, r, c, direction):
        nr, nc = self.get_neighbour_address(r, c, direction)
        return self.cells[nr][nc].get_out()

    def get_neighbour_flag(self, r, c, direction, flag):
        nr, nc = self.get_neighbour_address(r, c, direction)
        return self.cells[nr][nc].get_flag(flag)

    def load_direct(self, col, increment):
        return _dynamic_load_direct(self, col, increment)

    def store_direct(self, col, value, increment):
        _dynamic_store_direct(self, col, value, increment)

    def load_indirect(self, address):
        index = self.memory_index.get(address)
        return self.memory[index][1] if index is not None else -1

    def store_indirect(self, address, value):
        index = self.memory_index.get(address)
        if index is not None:
            self.memory[index][1] = value
            return
        self.memory_index[address] = len(self.memory)
        self.memory.append([address, value])

    def _this_pc_latency(self, pc):
        n_mem = n_mul = 0
        for row in pc.pe_ops:
            if len(row) != self.compute_cols:
                raise ValueError("Each PE instruction row must match compute_cols")
            for text in row:
                tokens = str(text or "").replace(",", " ").split()
                if tokens and tokens[0].upper() in ("LWD", "LWI", "SWD", "SWI"):
                    n_mem += 1
                if tokens and tokens[0].upper() == "SMUL":
                    n_mul += 1
        # Preserve cgra.py's established PE latency abstraction.  One LCU
        # ALU/branch operation takes one cycle and is co-issued with this PC.
        latency = max(1, 1 + n_mem if n_mem else 1, 3 if n_mul else 1)
        if n_mem:
            self.aproxMemCycles += 1 + n_mem
        else:
            self.aproxArithCycles += latency
        return latency

    def step(self):
        if self.exit:
            return True
        if self.instr2exec == len(self.instrs):
            self.exit = True
            return True
        if not 0 <= self.instr2exec < len(self.instrs):
            raise RuntimeError(f"PC {self.instr2exec} is outside the program")
        pc = self.instrs[self.instr2exec]
        if len(pc.pe_ops) != self.rows:
            raise ValueError("PE instruction matrix row count does not match architecture")
        lcu_ops = pc.lcu_ops if pc.lcu_ops is not None else [None] * self.rows
        if len(lcu_ops) != self.rows:
            raise ValueError("There must be one LCU command slot per row")
        if not self.architecture.lcu_per_row and any(lcu_ops):
            raise ValueError("This architecture profile has no LCU")

        for row in self.cells:
            for cell in row:
                cell.update()
        pe_targets = []
        for r in range(self.rows):
            for c in range(self.compute_cols):
                text = pc.pe_ops[r][c] or "NOP"
                opcode = str(text).replace(",", " ").split()
                is_branch = bool(opcode and opcode[0].upper() in
                                 ("BEQ", "BNE", "BLT", "BGE", "JUMP"))
                if is_branch and self.architecture.lcu_per_row:
                    raise ValueError("Loop/branch instructions belong in the LCU column")
                branch, exit_seen = self.cells[r][c].exec(text)
                if branch:
                    pe_targets.append(int(branch))
                self.exit = self.exit or bool(exit_seen)

        targets = []
        if self.architecture.lcu_per_row:
            for r, command in enumerate(lcu_ops):
                if command is not None and str(command).strip().upper() != "NOP":
                    self.lcu_operations += 1
                    if str(command).replace(",", " ").split()[0].upper() in (
                            "BEQ", "BNE", "BLT", "BGE", "JUMP"):
                        self.lcu_branch_operations += 1
                targets.append(self.lcus[r].execute(command, self.instr2exec))
            if any(command is not None and str(command).strip().upper() != "NOP"
                   for command in lcu_ops):
                self.lcu_issue_pcs += 1
        taken = [target for target in targets if target is not None]
        self.lcu_taken_branches += len(taken)
        if taken and (len(taken) != self.rows or len(set(taken)) != 1):
            raise RuntimeError(f"LCU branch divergence at PC {self.instr2exec}: {targets}")
        if pe_targets and self.architecture.lcu_per_row:
            raise RuntimeError("PE branches cannot control an LCU-equipped profile")
        if pe_targets and (len(set(pe_targets)) != 1 or taken):
            raise RuntimeError(f"Control branch conflict at PC {self.instr2exec}")

        latency = self._this_pc_latency(pc)
        if any(command is not None and str(command).strip().upper() != "NOP"
               for command in lcu_ops):
            self.aproxArithCycles += 0  # LCU work overlaps but remains cycle-charged by max().
        self.aproxcycles += latency
        self.cycles += 1
        self.instr2exec = (taken[0] if taken else
                           pe_targets[0] if pe_targets else self.instr2exec + 1)
        return self.exit

    def run(self, limit=1_000_000):
        while not self.step():
            if self.cycles > limit:
                raise RuntimeError(f"Execution limit reached ({limit} PCs)")
        config_cycles = len(self.instrs) * self.architecture.physical_cols
        return {
            "memory": self.memory,
            "execution_cycles": self.aproxcycles,
            "configuration_cycles": config_cycles,
            "total_cycles": self.aproxcycles + config_cycles,
            "memory_cycles": self.aproxMemCycles,
            "arithmetic_cycles": self.aproxArithCycles,
            "pcs_executed": self.cycles,
            "lcu_operations": self.lcu_operations,
            "lcu_branch_operations": self.lcu_branch_operations,
            "lcu_taken_branches": self.lcu_taken_branches,
            "lcu_issue_pcs": self.lcu_issue_pcs,
        }


def _dynamic_load_direct(parent, col, increment):
    index = parent.memory_index.get(parent.load_addr[col])
    value = parent.memory[index][1] if index is not None else -1
    parent.load_addr[col] += increment
    return value


def _dynamic_store_direct(parent, col, value, increment):
    index = parent.memory_index.get(parent.store_addr[col])
    if index is not None:
        parent.memory[index][1] = value
    else:
        parent.memory_index[parent.store_addr[col]] = len(parent.memory)
        parent.memory.append([parent.store_addr[col], value])
    parent.store_addr[col] += increment

class INSTR:
    def __init__( self,matrix):
        self.time = matrix[0][0]                        # ToDo: Fix how we assign this length
        self.ops = [[ matrix[r+1][c] for c in range(N_COLS)] for r in range(N_ROWS)]

def ker_parse( data ):
    instrs = int(len(data)/( INSTR_SIZE  )) # Always have a CSV with as many csv-columns as CGA-columns. Each instruction starts with the instruction timestamp i nthe first column. The next instruction must be immediately after the last row of this instruction.
    return [ INSTR( data[r_i*INSTR_SIZE:(r_i+1)*INSTR_SIZE][0:] ) for r_i in range(instrs) ]

def print_out( prs, outs, insts, ops, reg ):
    if PRINT_OUTS:
        out_string = ""

        if type(prs) == str:
            prs = [prs]

        for pr in prs:
            pnt = []
            if      pr == "ROUT" : pnt = outs
            elif    pr == "INST" : pnt = insts
            elif    pr == "OPS"  : pnt = ops
            elif    pr == "R0"   : pnt = reg[0]
            elif    pr == "R1"   : pnt = reg[1]
            elif    pr == "R2"   : pnt = reg[2]
            elif    pr == "R3"   : pnt = reg[3]

            out_string += "["
            for i in range(len(pnt)):
                out_string += "{{{}:4}}".format(i)
                if i == (len(pnt) - 1):
                    out_string += "]    "
                else:
                    out_string += ", "
            out_string = out_string.format(*[o for o in pnt])
        print(out_string)


class CGRA:
    def __init__( self, kernel, memory, read_addrs, write_addrs):
        self.cells = []
        for r in range(N_ROWS):
            list = []
            for c in range(N_COLS):
                list.append( PE( self, r, c) )
            self.cells.append(list)
        self.instrs     = ker_parse( kernel )
        self.memory     = memory
        self.memory_index = {}
        for index, row in enumerate(self.memory):
            self.memory_index.setdefault(row[0], index)
        self.instr2exec = 0
        self.cycles     = 0
        self.aproxcycles = 0
        self.aproxMemCycles = 0
        self.aproxArithCycles = 0
        if read_addrs is not None and len(read_addrs) == N_COLS: 
            self.load_addr = read_addrs
        else:   
            self.load_addr = [0]*N_COLS
        if write_addrs is not None and len(write_addrs) == N_COLS: 
            self.store_addr = write_addrs
        else:   
            self.store_addr = [0]*N_COLS
        self.exit       = False

    def run( self, pr, limit ):
        steps = 0
        while not self.step(pr):
            if (PRINT_OUTS):
                print("-------")
            steps += 1
            if steps > limit:
                print("EXECUTION LIMIT REACHED (",limit,"steps)")
                print("Extend the execution by calling the run with argument limit=<steps>.")
                return self.memory
        print("Execution accurate cycles: " + str(self.aproxcycles))
        config_cycles = len(self.instrs)*N_COLS
        print("Config cycles: " + str(config_cycles))
        print("Total cycles: " + str(config_cycles + self.aproxcycles))
        print("Memory cycles: " + str(self.aproxMemCycles))
        print("Arithmetic cycles: " + str(self.aproxArithCycles))

        return self.memory
    
    def maxCyclesThisPC(self, pc_index=None):
        nMemOps = 0
        nMulOps = 0
        # A taken branch updates ``self.instr2exec`` while the current PC is
        # being issued.  Latency still belongs to the PC that was just
        # executed, so callers pass its saved index explicitly.
        if pc_index is None:
            pc_index = self.instr2exec
        for r in range(N_ROWS):
            for c in range(N_COLS):
                instr =  (self.instrs[pc_index].ops[r][c]).replace(',', ' ').split()
                try:
                    op = instr[0]
                except:
                    op = instr
                if op in ['LWD','LWI','SWD','SWI']:
                    nMemOps += 1
                if op in ['SMUL']:
                    nMulOps += 1
        maxcycles = 1
        if nMemOps > 0:
            maxcycles = 1 + nMemOps
        if nMulOps > 0:
            maxcycles = max(maxcycles, 3)

        # Update arith/mem cycles
        if nMemOps > 0:
            self.aproxMemCycles += 1 + nMemOps
        else:
            self.aproxArithCycles += maxcycles
        
        return maxcycles
                    
    def step( self, prs="ROUT" ):
        for r in range(N_ROWS):
            for c in range(N_COLS):
                self.cells[r][c].update()
        instr2exec = self.instr2exec
        if PRINT_OUTS: print("Instr = ", self.cycles, "(",instr2exec,")")
        for r in range(N_ROWS):
            for c in range(N_COLS):
                op =  self.instrs[instr2exec].ops[r][c]
                b ,e = self.cells[r][c].exec( op )
                if b != 0: self.instr2exec = b - 1 #To avoid more logic afterwards
                if e != 0: self.exit = True
            outs    = [ self.cells[r][i].out        for i in range(N_COLS) ]
            insts   = [ self.cells[r][i].instr      for i in range(N_COLS) ]
            ops     = [ self.cells[r][i].op         for i in range(N_COLS) ]
            reg     = [[ self.cells[r][i].regs[regs[x]]   for i in range(N_COLS) ] for x in range(len(regs)) ]
            print_out( prs, outs, insts, ops, reg )

        ncyclesthisPc = self.maxCyclesThisPC(instr2exec)
        if PRINT_OUTS: print("Aprox cycles this pc: " + str(ncyclesthisPc))
        self.aproxcycles += ncyclesthisPc
        self.instr2exec += 1
        self.cycles += 1
        
        return self.exit

    def get_neighbour_address( self, r, c, dir ):
        n_r = r
        n_c = c
        if dir == "RCL": n_c = c - 1 if c > 0 else MAX_COL
        if dir == "RCR": n_c = c + 1 if c < MAX_COL else 0
        if dir == "RCT": n_r = r - 1 if r > 0 else MAX_ROW
        if dir == "RCB": n_r = r + 1 if r < MAX_ROW else 0
        return n_r, n_c

    def get_neighbour_out( self, r, c, dir ):
        n_r, n_c = self.get_neighbour_address( r, c, dir )
        return self.cells[n_r][n_c].get_out()

    def get_neighbour_flag( self, r, c, dir, flag ):
        n_r, n_c = self.get_neighbour_address( r, c, dir )
        return self.cells[n_r][n_c].get_flag( flag )

    def load_direct(self, c, incr):
        index = self.memory_index.get(self.load_addr[c])
        ret = self.memory[index][1] if index is not None else -1
        self.load_addr[c] += incr
        return ret

    def store_direct( self, c, val, incr ):
        index = self.memory_index.get(self.store_addr[c])
        if index is not None:
            self.memory[index][1] = val
        else:
            self.memory_index[self.store_addr[c]] = len(self.memory)
            self.memory.append([self.store_addr[c], val])
        self.store_addr[c] += incr
        return

    def load_indirect( self, addr ):
        index = self.memory_index.get(addr)
        return self.memory[index][1] if index is not None else -1

    def store_indirect( self, addr, val):
        index = self.memory_index.get(addr)
        if index is not None:
            self.memory[index][1] = val
            return
        self.memory_index[addr] = len(self.memory)
        self.memory.append([addr, val])
        return

class PE:
    def __init__( self, parent, row, col ):
        self.parent = parent
        self.row = row
        self.col = col
        self.flags      = { "sign"   : 0,
                            "zero"   : 0,
                            "branch" : 0,
                            "exit"   : 0}
        self.instr      = ""
        self.old_out    = 0
        self.out        = 0
        self.regs       = {'R0':0, 'R1':0, 'R2':0, 'R3':0 }
        self.op         = ""
        self.instr      = ""

    def get_out( self ):
        return self.old_out

    def get_flag( self, flag ):
        return self.flags[flag]

    def fetch_val( self, val):
        if val.lstrip('-+').isnumeric():
            return int(val)
        if val == 'ROUT':
            return int( self.old_out)
        if val == 'ZERO':
            return 0
        if val in self.regs:
            return int( self.regs[val])
        return int(self.parent.get_neighbour_out( self.row, self.col, val ))

    def fetch_flag( self, dir, flag ):
        if dir == 'ROUT':
            return int( self.old_out)
        return int(self.parent.get_neighbour_flag( self.row, self.col, dir, flag ))

    def exec( self,  instr ):
        self.run_instr(instr)
        return self.flags["branch"], self.flags["exit"]

    def update( self):
        self.old_out = self.out
        self.flags["zero"]      = 1 if self.out == 0 else 0
        self.flags["sign"]      = 1 if self.out <  0 else 0
        self.flags["branch"]    = 0

    def run_instr( self, instr):
        instr   = instr.replace(',', ' ')   # Remove the commas so we can speparate arguments by spaces
        self.instr = instr                  # Save this string as instruction to show
        instr   = instr.split()             # Split into chunks
        try:
            self.op      = instr[0]
        except:
            self.op = instr

        if self.op in self.ops_arith:
            des     = instr[1]
            val1    = self.fetch_val( instr[2] )
            val2    = self.fetch_val( instr[3] )
            ret     = self.ops_arith[self.op]( val1, val2)
            #if (self.op == 'SMUL'):
            #    print("ret = ", ret, ", val1 = ", val1, ", val2 = ", val2, ", des = ", des, ", instr[2] =", instr[2], ", instr[3] =", instr[3] )
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_cond:
            des     = instr[1]
            val1    = self.fetch_val( instr[2] )
            val2    = self.fetch_val( instr[3] )
            src     = instr[4]
            method  = self.ops_cond[self.op]
            ret     = method(self, val1, val2, src)
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_branch:
            val1    = self.fetch_val( instr[1] )
            val2    = self.fetch_val( instr[2] )
            branch  = self.fetch_val( instr[3] )
            method = self.ops_branch[self.op]
            method(self, val1, val2, branch)
            self.out = branch

        elif self.op in self.ops_lwd:
            des = instr[1]
            incr = self.fetch_val(instr[2])
            ret = self.parent.load_direct( self.col, incr )
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_swd:
            val = self.fetch_val( instr[1] )
            incr = self.fetch_val(instr[2])
            self.parent.store_direct( self.col, val, incr )

        elif self.op in self.ops_lwi:
            des = instr[1]
            addr = self.fetch_val( instr[2] )
            ret = self.parent.load_indirect(addr)
            if des in self.regs: self.regs[des] = ret
            self.out = ret

        elif self.op in self.ops_swi:
            addr = self.fetch_val( instr[2] )
            val = self.fetch_val( instr[1] )
            self.parent.store_indirect( addr, val )
            pass

        elif self.op in self.ops_nop:
            pass # Intentional

        elif self.op in self.ops_jump:
            val1    = self.fetch_val( instr[1] )
            val2    = self.fetch_val( instr[2] )
            self.jump( val1 + val2)
            self.out = self.out # Intentional ROUT not modified

        elif self.op in self.ops_exit:
            self.flags['exit'] = 1

    def sadd( val1, val2 ):
        return c_int32( val1 + val2 ).value

    def ssub( val1, val2 ):
        return c_int32( val1 - val2 ).value

    def smul( val1, val2 ):
        return c_int32( val1 * val2 ).value

    def fxpmul( val1, val2 ):
        print("Better luck next time")
        return 0

    def slt( val1, val2 ):
        return c_int32(val1 << val2).value
    
    def sll( val1, val2 ):
        return c_int32(val1 << val2).value

    def srt( val1, val2 ):
        interm_result = (c_int32(val1).value & MAX_32b)
        return c_int32(interm_result >> val2).value
    
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
        return c_int32( ((val1& MAX_32b) ^ (val2& MAX_32b)) & MAX_32b).value

    def lnand( val1, val2 ):
        return c_int32( ~( val1 & val2 ) & MAX_32b ).value

    def lnor( val1, val2 ):
         return c_int32( ~( val1 | val2 ) & MAX_32b ).value

    def lxnor( val1, val2 ):
        return c_int32( ~( val1 ^ val2 ) & MAX_32b ).value

    def bsfa( self, val1, val2, src):
        flag = self.fetch_flag( src, 'sign')
        return val1 if flag == 1 else val2

    def bzfa( self,  val1, val2, src):
        flag = self.fetch_flag( src, 'zero')
        return val1 if flag == 1 else val2

    def beq( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 == val2 else self.flags['branch']

    def bne( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 != val2 else self.flags['branch']

    def bge( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 >= val2 else self.flags['branch']

    def blt( self,  val1, val2, branch ):
        self.flags['branch'] = branch if val1 < val2 else self.flags['branch']
    
    def jump( self, branch):
        self.flags['branch'] = branch

    ops_arith   = { 'SADD'      : sadd,
                    'SSUB'      : ssub,
                    'SMUL'      : smul,
                    'FXPMUL'    : fxpmul,
                    'SLT'       : slt,
                    'SLL'       : sll,
                    'SRT'       : srt,
                    'SRL'       : srl,
                    'SRA'       : sra,
                    'LOR'       : lor,
                    'LAND'      : land,
                    'LXOR'      : lxor,
                    'LNAND'     : lnand,
                    'LNOR'      : lnor,
                    'LXNOR'     : lxnor }

    ops_cond    = { 'BSFA'      : bsfa,
                    'BZFA'      : bzfa }

    ops_branch  = { 'BEQ'       : beq,
                    'BNE'       : bne,
                    'BLT'       : blt,
                    'BGE'       : bge }

    ops_lwd     = { 'LWD'       : '' }
    ops_swd     = { 'SWD'       : '' }
    ops_lwi     = { 'LWI'       : '' }
    ops_swi     = { 'SWI'       : '' }

    ops_nop     = { 'NOP'       : '' }
    ops_jump    = { 'JUMP'      : '' }
    ops_exit    = { 'EXIT'      : '' }

def estimatedConfigCycles( kernel, version=""):
    ker = []
    # Read the instructions file
    with open( kernel + "/"+FILENAME_INSTR+version+EXT, 'r') as f:
        for row in csv.reader(f): ker.append(row)
    instrs = ker_parse(ker)
    return len(instrs)*N_COLS


def run( kernel, version="", pr="ROUT", limit=100, load_addrs=None, store_addrs=None, printVal=1 ):
    global PRINT_OUTS
    PRINT_OUTS = printVal
    ker = []
    mem = []

    # Read the instructions file
    path_completo = os.path.join(kernel, f"{FILENAME_INSTR}{version}{EXT}")

    with open(path_completo, 'r') as f:
        for row in csv.reader(f):
            ker.append(row)
    
    # Create an empty memory file if there is not any
    if not os.path.isfile(kernel + "/"+FILENAME_MEM+version+EXT):
        kernel_clear_memory(kernel, version)

    # Read the memory file
    with open( kernel + "/"+FILENAME_MEM+version+EXT, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        # Skip the first row (header)
        next(csv_reader, None)
        for row in csv_reader:
            try:
                mem.append([int(row[0]), int(row[1])])
            except ValueError:
                print("Error: Values in CSV file are not integers.")
                return None
    
    # Run the kernel
    cgra = CGRA(ker, mem, load_addrs, store_addrs)
    mem = cgra.run(pr, limit)

    # Store the output sorted
    sorted_mem = sorted(mem, key=lambda x: x[0])
    with open( kernel + "/"+FILENAME_MEM_O+version+EXT, 'w+') as f:
        for row in sorted_mem: csv.writer(f).writerow(row)

    print("END")

