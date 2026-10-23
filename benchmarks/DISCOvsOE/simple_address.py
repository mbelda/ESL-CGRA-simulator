"""Reproducible Simple Address ablation for the original MMUL and GEM kernels.

The Python preparation pass traces the original indirect accesses, then lays
out each column's exact load stream and output stream contiguously. The CGRA
still performs every arithmetic operation and all loop control.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cgra as cgra_module
from cgra import CGRA, PE

cgra_module.PRINT_OUTS = 0
ROWS = COLS = 4
START_ADDR = 20000
STREAM_BASE = 1_000_000_000


def instructions(path):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return [rows[i:i + 5] for i in range(0, len(rows), 5)]


def cfg_mmul(m, k, n, A, B, C):
    a = START_ADDR
    b = a + len(A) * 4
    c = b + len(B) * 4
    nc, nr = n // 4, m // 4
    cfg = [
        [b, nc, a + 2*k*4, c + 3*n*4, -4*n, k],
        [c + 4, b + 4, k, a + 3*k*4, k, -4*n],
        [a, c + 2*4 + n*4, b + 2*4, nc, k, -4*n],
        [nr, a + k*4, c + 3*4 + 2*n*4, b + 3*4, k, -4*n],
    ]
    return cfg, a, b, c


def cfg_gem(m, k, n, A, B, C, alpha, beta):
    a = START_ADDR
    b = a + len(A) * 4
    c = b + len(B) * 4
    nc, nr = n // 4, m // 4
    cfg = [
        [b, nc, a + 2*k*4, c + 3*n*4, -4*n, beta, k],
        [c + 4, b + 4, k, a + 3*k*4, k, -4*n, beta],
        [a, c + 2*4 + n*4, b + 2*4, nc, alpha, k, -4*n],
        [nr, a + k*4, c + 3*4 + 2*n*4, b + 3*4, alpha, k, -4*n],
    ]
    return cfg, a, b, c


def memory_from(cfg, a, b, c, A, B, C):
    mem = []
    p = 0
    for col in cfg:
        mem.extend([[p + 4*i, v] for i, v in enumerate(col)])
        p += 4 * len(col)
    mem.extend([[a + 4*i, v] for i, v in enumerate(A)])
    mem.extend([[b + 4*i, v] for i, v in enumerate(B)])
    mem.extend([[c + 4*i, v] for i, v in enumerate(C)])
    return mem, [0, len(cfg[0])*4, (len(cfg[0])+len(cfg[1]))*4,
                 (len(cfg[0])+len(cfg[1])+len(cfg[2]))*4]


def accelerated_memory(c):
    """Use equivalent indexed lookup in both modes; preserve first duplicate."""
    values = {}
    for addr, val in c.memory:
        values.setdefault(addr, val)
    c._memory_index = values
    positions = {}
    for i, (addr, _) in enumerate(c.memory):
        positions.setdefault(addr, i)
    c._memory_positions = positions
    def ld(addr):
        return c._memory_index.get(addr, -1)
    def st(addr, val):
        if addr not in c._memory_positions:
            c.memory.append([addr, val])
            c._memory_positions[addr] = len(c.memory) - 1
        else:
            c.memory[c._memory_positions[addr]][1] = val
        c._memory_index[addr] = val
    def direct_load(col, incr):
        addr = c.load_addr[col]
        val = c._memory_index.get(addr, -1)
        c.load_addr[col] += incr
        return val
    def direct_store(col, val, incr):
        addr = c.store_addr[col]
        st(addr, val)
        c.store_addr[col] += incr
    c.load_indirect = ld
    c.store_indirect = st
    c.load_direct = direct_load
    c.store_direct = direct_store


def run(kernel, mem, loads, stores=None, trace=False, limit=20_000_000):
    if kernel and isinstance(kernel[0], list) and len(kernel[0]) == 5:
        kernel = flatten_program(kernel)
    c = CGRA(kernel, mem, loads, stores or [0]*4)
    accelerated_memory(c)
    accesses = {"loads": [[] for _ in range(4)], "stores": [[] for _ in range(4)]}
    direct = {"loads": [[] for _ in range(4)], "stores": [[] for _ in range(4)]}
    old_direct_load, old_direct_store = c.load_direct, c.store_direct
    def logged_load(col, incr):
        addr = c.load_addr[col]
        val = old_direct_load(col, incr)
        direct['loads'][col].append((addr, val))
        return val
    def logged_store(col, val, incr):
        addr = c.store_addr[col]
        old_direct_store(col, val, incr)
        direct['stores'][col].append((addr, val))
    c.load_direct, c.store_direct = logged_load, logged_store
    old = PE.run_instr
    def traced(pe, text):
        if trace:
            fields = text.replace(',', ' ').split()
            if fields and fields[0] == 'LWI':
                addr = pe.fetch_val(fields[2])
                accesses['loads'][pe.col].append((addr, c._memory_index.get(addr, -1)))
            elif fields and fields[0] == 'SWI':
                addr = pe.fetch_val(fields[2])
                accesses['stores'][pe.col].append(addr)
        return old(pe, text)
    PE.run_instr = traced
    try:
        steps = 0
        while not c.step():
            steps += 1
            if steps > limit:
                raise RuntimeError(f"instruction limit exceeded ({limit})")
    finally:
        PE.run_instr = old
    return c, accesses, direct


def flatten_program(chunks):
    return [row for chunk in chunks for row in chunk]


def transform(source, kind):
    """Replace indirect accesses and compact address-only instructions/PCs."""
    pcs = [[r[:] for r in pc] for pc in source]
    # Direct stream accesses advance one word after each access.
    for pc in pcs:
        for row in range(1, 5):
            for col in range(4):
                op = pc[row][col].replace(',', ' ').split()
                if op and op[0] == 'LWI':
                    pc[row][col] = f"LWD {op[1]}, 4"
                elif op and op[0] == 'SWI':
                    pc[row][col] = f"SWD {op[1]}, 4"

    address_pcs = ({8, 9, 10, 11, 13, 14, 15, 17, 18, 19} if kind == 'mmul'
                   else {8, 9, 10, 11, 22, 23, 25, 26, 27, 28, 29, 30})
    suppressed_ops = 0
    suppressed_pc_ids = set()
    for pc_id in address_pcs:
        for r in range(1, 5):
            for col in range(4):
                text = pcs[pc_id][r][col]
                op = text.replace(',', ' ').split()
                if not op or op[0] == 'NOP' or op[0] in ('BNE', 'BEQ', 'BLT', 'BGE'):
                    continue
                # Keep the local loop counters, accumulator clears, and
                # explicit counter broadcasts; discard pointer arithmetic.
                keep = (op[0] == 'SADD' and
                        (op[1] in ('R1', 'R3') or
                         (op[1] == 'ROUT' and len(op) > 3 and op[3] == 'R1')))
                if not keep:
                    pcs[pc_id][r][col] = 'NOP'
                    suppressed_ops += 1
                    suppressed_pc_ids.add(pc_id)

    # PCs that became entirely empty only carried address generation.
    keep_ids = [i for i, pc in enumerate(pcs)
                if any(pc[r][col].replace(',', ' ').split()[0] != 'NOP'
                       for r in range(1, 5) for col in range(4))]
    pc_map = {old: new for new, old in enumerate(keep_ids)}
    for new, old in enumerate(keep_ids):
        pcs[old][0][0] = str(new)

    # Remap absolute PC targets used by BNE to their compacted position.
    for old_pc in keep_ids:
        pc = pcs[old_pc]
        for r in range(1, 5):
            for col in range(4):
                op = pc[r][col].replace(',', ' ').split()
                if op and op[0] in ('BNE', 'BEQ', 'BLT', 'BGE'):
                    target = int(op[3])
                    pc[r][col] = f"{op[0]} {op[1]}, {op[2]}, {pc_map[target]}"
    return [pcs[i] for i in keep_ids], keep_ids, suppressed_ops, sorted(suppressed_pc_ids)


def save_instructions(path, program):
    with path.open('w', newline='') as f:
        csv.writer(f).writerows(flatten_program(program))


def reference(kind, A, B, C, m, k, n, alpha=1, beta=0):
    out = []
    for i in range(m):
        for j in range(n):
            v = sum(A[i*k+x] * B[x*n+j] for x in range(k))
            out.append(alpha*v + beta*C[i*n+j])
    return out


def experiment(kind, size, seed=12345):
    m = k = n = size
    rng = random.Random(seed)
    A = [rng.randint(-10, 10) for _ in range(m*k)]
    B = [rng.randint(-10, 10) for _ in range(k*n)]
    C0 = [rng.randint(-10, 10) for _ in range(m*n)]
    alpha, beta = 125, 14
    src_path = ROOT / ('benchmarks/DISCOvsOE/mmul/instructions.csv' if kind == 'mmul'
                       else 'benchmarks/gemm/instructions_meth_compact.csv')
    src = instructions(src_path)
    config, a, b, caddr = (cfg_mmul(m,k,n,A,B,C0) if kind == 'mmul'
                           else cfg_gem(m,k,n,A,B,C0,alpha,beta))
    mem0, load0 = memory_from(config,a,b,caddr,A,B,C0)
    base, _, _ = run(src, mem0, load0, trace=False)
    expected = reference(kind,A,B,C0,m,k,n,alpha if kind=='gem' else 1,beta if kind=='gem' else 0)
    # Baseline output from conventional layout.
    original = [base._memory_index.get(caddr+4*i, 0) for i in range(m*n)]
    if original != expected:
        raise AssertionError(f'{kind} original reference mismatch at {size}: {sum(x!=y for x,y in zip(original,expected))}')
    base_cycles = base.aproxcycles + len(base.instrs)*4

    # Trace original indirect addresses and write ordering to create static streams.
    prep_start = time.perf_counter()
    mem1, load1 = memory_from(config,a,b,caddr,A,B,C0)
    _, trace, _ = run(src, mem1, load1, trace=True)
    transformed, retained, suppressed_ops, suppressed_pc_ids = transform(src, kind)
    output_dir = ROOT / ('benchmarks/DISCOvsOE/mmul' if kind == 'mmul'
                         else 'benchmarks/DISCOvsOE/gem')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_instructions(output_dir / 'instructions_simple_address.csv', transformed)
    simple_mem = []
    load_addrs = []
    cfg_starts = []
    cursor = STREAM_BASE
    for col in range(4):
        vals = list(config[col]) + [val for _, val in trace['loads'][col]]
        cfg_starts.append(cursor)
        simple_mem.extend([[cursor + 4*i, v] for i,v in enumerate(vals)])
        load_addrs.append(cursor)
        cursor += 4*len(vals) + 64
    write_addrs = []
    store_map = [[] for _ in range(4)]
    for col in range(4):
        write_addrs.append(cursor)
        for i, logical_addr in enumerate(trace['stores'][col]):
            store_map[col].append((cursor + 4*i, logical_addr))
            simple_mem.append([cursor + 4*i, 0])
        cursor += 4*len(trace['stores'][col]) + 64
    prep_seconds = time.perf_counter() - prep_start
    # Assert direct stream order and uniqueness of the store destinations.
    expected_stores = [addr for col in trace['stores'] for addr in col]
    if len(expected_stores) != len(set(expected_stores)):
        raise AssertionError('baseline store trace has repeated output locations')
    simple, _, direct = run(transformed, simple_mem, load_addrs, write_addrs)
    load_trace_valid = True
    for col in range(4):
        direct_values = [v for _, v in direct['loads'][col]][len(config[col]):]
        expected_values = [v for _, v in trace['loads'][col]]
        load_trace_valid &= direct_values == expected_values
    store_trace_valid = True
    for col in range(4):
        observed_addresses = [addr for addr, _ in direct['stores'][col]]
        planned_addresses = [addr for addr, _ in store_map[col]]
        store_trace_valid &= observed_addresses == planned_addresses
    if not load_trace_valid or not store_trace_valid:
        raise AssertionError('direct access trace differs from prepared sequence')
    result = [0] * (m*n)
    for col in range(4):
        for stream_addr, logical_addr in store_map[col]:
            value = simple.memory[simple._memory_positions[stream_addr]][1]
            result[(logical_addr-caddr)//4] = value
    if result != expected:
        mismatch = [(i,x,y) for i,(x,y) in enumerate(zip(result,expected)) if x!=y][:5]
        raise AssertionError(f'{kind} simple reference mismatch {size}: {mismatch}')
    simple_cycles = simple.aproxcycles + len(simple.instrs)*4
    unique_loaded = len({addr for col in trace['loads'] for addr,_ in col})
    repeated_loads = sum(map(len, trace['loads'])) - unique_loaded
    original_words = len(mem0)
    simple_words = len(simple_mem)
    return {
        'benchmark': kind, 'size': size, 'seed': seed,
        'original_cycles': base_cycles, 'simple_cycles': simple_cycles,
        'reduction_percent': 100*(base_cycles-simple_cycles)/base_cycles,
        'speedup': base_cycles/simple_cycles,
        'correct': True, 'original_instructions': len(src),
        'simple_instructions': len(transformed), 'retained_pc_ids': retained,
        'removed_pc_ids': sorted(set(range(len(src)))-set(retained)),
        'suppressed_address_ops': suppressed_ops,
        'suppressed_address_pc_ids': suppressed_pc_ids,
        'indirect_load_accesses': sum(map(len,trace['loads'])),
        'indirect_store_accesses': sum(map(len,trace['stores'])),
        'unique_indirect_load_addresses': unique_loaded,
        'repeated_load_values': repeated_loads,
        'original_memory_words': original_words,
        'simple_memory_words': simple_words,
        'memory_factor': simple_words/original_words,
        'preparation_seconds': prep_seconds,
        'direct_load_trace_validated': bool(load_trace_valid),
        'direct_store_trace_validated': bool(store_trace_valid),
        'original_memory_payload_bytes': original_words * 4,
        'simple_memory_payload_bytes': simple_words * 4,
        'additional_memory_payload_bytes': (simple_words-original_words) * 4,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', choices=['mmul','gem','all'], default='all')
    p.add_argument('--sizes', nargs='+', type=int, default=[16,32,64,128])
    p.add_argument('--output', default=str(ROOT/'benchmarks/DISCOvsOE/simple_address_results.json'))
    args = p.parse_args()
    results = []
    for kind in (['mmul','gem'] if args.benchmark=='all' else [args.benchmark]):
        for size in args.sizes:
            print(f'Running {kind} {size}x{size}', flush=True)
            results.append(experiment(kind,size))
            print(json.dumps(results[-1], sort_keys=True), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
