#!/usr/bin/env python3
"""Run a reproducible Simple Address + LCU ablation on MMUL and GEM.

The Simple Address instruction files are imported unchanged from the validated
ablation.  For an LCU run, a PE PC containing only a conditional branch is
removed and its LCU operation is issued at the end of the preceding PE PC.
Mixed branch/data PCs stay in place and use the established co-issued LCU path.
All direct-memory streams are generated from an indirect-address trace of the
original kernel and checked against the executed direct-access trace.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DISCO = ROOT / "benchmarks/DISCOvsOE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DISCO))

import cgra
import simple_address as sa
from heterogeneous import run_preserved_mesh as lcu_ref

cgra.PRINT_OUTS = 0

SIZES = (16, 32, 64, 128)
SEED = 12345
DIRECT_BASE = 1_000_000_000


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def pc_chunks(rows):
    if len(rows) % 5:
        raise ValueError(f"Instruction CSV has {len(rows)} rows, not a multiple of 5")
    return [[list(row) for row in rows[i:i + 5]] for i in range(0, len(rows), 5)]


def flatten(chunks):
    return [list(row) for pc in chunks for row in pc]


def branch_rows(chunks):
    found = []
    for pc, chunk in enumerate(chunks):
        for r in range(1, 5):
            for col in range(4):
                words = chunk[r][col].replace(",", " ").split()
                if words and words[0].upper() in ("BEQ", "BNE", "BLT", "BGE"):
                    if len(words) != 4:
                        raise ValueError(f"Malformed branch at PC {pc}, RC({r-1},{col})")
                    found.append(lcu_ref.Branch(pc, r - 1, col, words[0].upper(),
                                                words[1], words[2], words[3]))
    return found


def compile_lcu_program(rows):
    """Compact branch-only PCs and remap all remaining branch targets.

    A branch-only PC is evaluated after its immediately preceding PC. This
    preserves the loop-register values visible to the original branch while
    removing one otherwise empty issue cycle. Its Rout target sideband is
    written before the next PC, matching the old branch PC's output.
    """
    chunks = pc_chunks(rows)
    branches = branch_rows(chunks)
    by_pc = defaultdict(list)
    for branch in branches:
        by_pc[branch.pc].append(branch)
    for pc, found in by_pc.items():
        if len(found) > 1:
            raise ValueError(f"Multiple branches in PC {pc} are unsupported")

    removed = set()
    for pc, found in by_pc.items():
        branch = found[0]
        nonbranch_count = 0
        for r in range(1, 5):
            for col in range(4):
                words = chunks[pc][r][col].replace(",", " ").split()
                if not words or words[0].upper() == "NOP":
                    continue
                if r - 1 == branch.row and col == branch.col and words[0].upper() == branch.op:
                    continue
                nonbranch_count += 1
        if nonbranch_count == 0:
            removed.add(pc)

    keep = [pc for pc in range(len(chunks)) if pc not in removed]
    pc_map = {old: new for new, old in enumerate(keep)}
    moved = defaultdict(list)
    new_chunks = [chunks[old] for old in keep]

    for old_pc in keep:
        pc = chunks[old_pc]
        for r in range(1, 5):
            for col in range(4):
                words = pc[r][col].replace(",", " ").split()
                if words and words[0].upper() in ("BEQ", "BNE", "BLT", "BGE"):
                    target = int(words[3], 0)
                    if target not in pc_map:
                        raise ValueError(f"Branch target {target} points to removed PC")
                    pc[r][col] = f"{words[0]} {words[1]}, {words[2]}, {pc_map[target]}"

    for old_pc in sorted(removed):
        previous = old_pc - 1
        if previous not in pc_map:
            raise ValueError(f"Cannot relocate branch at first PC {old_pc}")
        branch = by_pc[old_pc][0]
        # The LCU reads the condition at the *start* of the preceding PC.
        # Reject a compaction that would need a same-PC PE-register bypass.
        operands = {branch.lhs, branch.rhs, branch.target}
        if any(token not in ("R0", "R1", "R2", "R3", "ZERO") and
               not token.lstrip("-+").isdecimal() for token in operands):
            raise ValueError(f"Relocated branch at PC {old_pc} needs a nonlocal operand")
        previous_words = chunks[previous][branch.row + 1][branch.col].replace(",", " ").split()
        previous_writes_register = (len(previous_words) > 1 and previous_words[0].upper() not in
                                    ("NOP", "EXIT", "SWD", "SWI", "BNE", "BEQ", "BLT", "BGE", "JUMP"))
        if previous_writes_register and previous_words[1] in operands:
            raise ValueError(f"Relocated branch at PC {old_pc} needs same-PC forwarding")
        target = int(branch.target, 0)
        if target not in pc_map:
            raise ValueError(f"Branch target {target} points to removed PC")
        moved_pc = pc_map[previous]
        moved[moved_pc].append(lcu_ref.Branch(
            branch.pc, branch.row, branch.col, branch.op, branch.lhs,
            branch.rhs, str(pc_map[target])))
        # Branch slots disappear with their containing PC; all retained
        # branches stay in the PE CSV until ControlCompareCGRA co-issues them.

    for new_pc, chunk in enumerate(new_chunks):
        chunk[0][0] = str(new_pc)
    def noncontrol_ops(program):
        return Counter(
            text.strip() for chunk in program for row in chunk[1:] for text in row
            if text.strip() and text.replace(",", " ").split()[0].upper()
            not in ("NOP", "BEQ", "BNE", "BLT", "BGE", "JUMP")
        )
    if noncontrol_ops(chunks) != noncontrol_ops(new_chunks):
        raise AssertionError("LCU compaction changed a noncontrol PE operation")
    return flatten(new_chunks), dict(moved), len(branches), sorted(removed), pc_map


class RelocatedControlCGRA(lcu_ref.ControlCompareCGRA):
    """ControlCompareCGRA with branch-only LCU ops moved to a prior PE PC."""

    def __init__(self, *args, relocated_branches=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.relocated_branches = relocated_branches or {}

    def step(self, prs="ROUT"):
        current_pc = self.instr2exec
        moved = []
        for branch in self.relocated_branches.get(current_pc, ()):
            source = self.cells[branch.row][branch.col]
            moved.append((branch, source.fetch_val(branch.lhs),
                          source.fetch_val(branch.rhs), source.fetch_val(branch.target)))
        done = super().step(prs)
        for branch, lhs, rhs, target in moved:
            if not self.lcus:
                raise RuntimeError("Relocated control requires at least one LCU")
            source = self.cells[branch.row][branch.col]
            taken = lcu_ref.compare(branch.op, lhs, rhs)
            self.branch_visits[branch.pc] += 1
            if taken:
                self.branch_taken[branch.pc] += 1
            targets = []
            for lcu in self.lcus:
                lcu.regs["R0"] = lhs
                lcu.regs["R1"] = rhs
                self.lcu_operation_count += 1
                self.lcu_branch_count += 1
                self.lcu_operations_by_unit[str(lcu.row)] += 1
                targets.append(lcu.execute(
                    f"{branch.op} R0 R1 {target}", current_pc))
            expected_target = target if taken else None
            if any(value != expected_target for value in targets):
                raise RuntimeError(f"LCU disagreement at relocated branch: {targets}")
            self.lcu_issue_cycles += 1
            # The removed branch PC would have written its target into Rout;
            # reproduce that value before the next PE PC updates old_out.
            source.out = target
            if taken:
                self.instr2exec = target
            self.control_digest.update(repr((branch.pc, branch.row, branch.col,
                                             branch.op, lhs, rhs, target, taken)).encode("ascii"))
        return done


def matrices(n):
    rng = random.Random(SEED)
    return ([rng.randint(-10, 10) for _ in range(n * n)],
            [rng.randint(-10, 10) for _ in range(n * n)],
            [rng.randint(-10, 10) for _ in range(n * n)])


def normalized_rows(path):
    return [[value.strip() for value in row] for row in read_rows(path)]


def prepare_simple_stream(benchmark, n, a, b, c0):
    if benchmark == "MMUL":
        config, a_addr, b_addr, c_addr = sa.cfg_mmul(n, n, n, a, b, c0)
        source_path = DISCO / "mmul/instructions.csv"
        simple_path = DISCO / "mmul/instructions_simple_address.csv"
    else:
        config, a_addr, b_addr, c_addr = sa.cfg_gem(n, n, n, a, b, c0, 125, 14)
        source_path = ROOT / "benchmarks/gemm/instructions_meth_compact.csv"
        simple_path = DISCO / "gem/instructions_simple_address.csv"

    current_baseline = lcu_ref.BENCHMARK_FILES[benchmark]
    if normalized_rows(source_path) != normalized_rows(current_baseline):
        raise RuntimeError(f"The Simple Address source differs from the current {benchmark} baseline")

    source = sa.instructions(source_path)
    baseline_memory, baseline_loads = sa.memory_from(config, a_addr, b_addr, c_addr, a, b, c0)
    baseline_sim, access_trace, _ = sa.run(source, baseline_memory, baseline_loads, trace=True)
    expected = lcu_ref.reference(a, b, c0, n, benchmark == "GEM")
    observed_baseline = [baseline_sim._memory_index.get(c_addr + 4 * i, 0)
                         for i in range(n * n)]
    if observed_baseline != expected:
        raise AssertionError(f"Original {benchmark} does not match the reference at N={n}")

    memory = []
    load_addrs = []
    cursor = DIRECT_BASE
    for col in range(4):
        config_and_data = list(config[col]) + [value for _, value in access_trace["loads"][col]]
        load_addrs.append(cursor)
        memory.extend([[cursor + 4 * i, value] for i, value in enumerate(config_and_data)])
        cursor += 4 * len(config_and_data) + 64

    store_addrs = []
    store_map = [[] for _ in range(4)]
    for col in range(4):
        store_addrs.append(cursor)
        for i, logical_addr in enumerate(access_trace["stores"][col]):
            stream_addr = cursor + 4 * i
            store_map[col].append((stream_addr, logical_addr))
            memory.append([stream_addr, 0])
        cursor += 4 * len(store_map[col]) + 64

    all_logical_stores = [addr for col in access_trace["stores"] for addr in col]
    if len(all_logical_stores) != n * n or len(set(all_logical_stores)) != n * n:
        raise AssertionError("Original store trace is not a one-to-one traversal of C")
    return {
        "config": config, "c_addr": c_addr, "expected": expected,
        "memory": memory, "load_addrs": load_addrs, "store_addrs": store_addrs,
        "store_map": store_map, "baseline_access_trace": access_trace,
        "simple_rows": read_rows(simple_path), "simple_path": simple_path,
    }


def run_simple_case(benchmark, n, mode, stream):
    lcu_enabled = mode != "OE_4x4"
    rows = stream["simple_rows"]
    instruction_path = stream["simple_path"]
    relocated = {}
    removed_pcs = []
    lcu_branches = 0
    pc_map = {pc: pc for pc in range(len(rows) // 5)}
    if lcu_enabled:
        rows, relocated, lcu_branches, removed_pcs, pc_map = compile_lcu_program(rows)
        if not removed_pcs:
            raise RuntimeError(f"Expected branch-only PCs after Simple Address for {benchmark}")
        out_name = f"{benchmark.lower()}_instructions_simple_address_lcu.csv"
        instruction_path = HERE / out_name
        with instruction_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        rows = read_rows(instruction_path)

    sim_class = RelocatedControlCGRA if lcu_enabled else lcu_ref.ControlCompareCGRA
    kwargs = {"relocated_branches": relocated} if lcu_enabled else {}
    sim = sim_class(rows, stream["memory"], list(stream["load_addrs"]), mode,
                    trace=False, **kwargs)
    sim.store_addr = list(stream["store_addrs"])

    direct = {"loads": [[] for _ in range(4)], "stores": [[] for _ in range(4)]}
    old_load, old_store = sim.load_direct, sim.store_direct

    def logged_load(col, increment):
        address = sim.load_addr[col]
        value = old_load(col, increment)
        direct["loads"][col].append((address, value))
        return value

    def logged_store(col, value, increment):
        address = sim.store_addr[col]
        old_store(col, value, increment)
        direct["stores"][col].append((address, value))

    sim.load_direct = logged_load
    sim.store_direct = logged_store
    with contextlib.redirect_stdout(io.StringIO()):
        final_memory = sim.run(pr=[], limit=20_000_000)

    expected_load_values = [
        list(stream["config"][col]) +
        [value for _, value in stream["baseline_access_trace"]["loads"][col]]
        for col in range(4)
    ]
    load_ok = True
    for col in range(4):
        observed = direct["loads"][col]
        expected_values = expected_load_values[col]
        planned_addresses = [stream["load_addrs"][col] + 4 * i
                             for i in range(len(expected_values))]
        load_ok &= ([value for _, value in observed] == expected_values and
                    [address for address, _ in observed] == planned_addresses)

    store_ok = True
    result = [None] * (n * n)
    final_values = {address: value for address, value in final_memory}
    for col in range(4):
        observed = direct["stores"][col]
        planned = stream["store_map"][col]
        observed_addresses = [address for address, _ in observed]
        expected_addresses = [address for address, _ in planned]
        store_ok &= observed_addresses == expected_addresses
        for (stream_addr, logical_addr), (_, value) in zip(planned, observed):
            committed_value = final_values.get(stream_addr)
            store_ok &= stream_addr in final_values and committed_value == value
            result[(logical_addr - stream["c_addr"]) // 4] = committed_value

    load_pointer_ok = all(
        sim.load_addr[col] == stream["load_addrs"][col] + 4 * len(direct["loads"][col])
        for col in range(4))
    store_pointer_ok = all(
        sim.store_addr[col] == stream["store_addrs"][col] + 4 * len(direct["stores"][col])
        for col in range(4))

    all_direct_inputs_valid = bool(load_ok and store_ok and load_pointer_ok and store_pointer_ok)
    correct = result == stream["expected"]
    if not all_direct_inputs_valid:
        load_diffs = {}
        store_diffs = {}
        for col in range(4):
            got_loads = direct["loads"][col]
            want_values = expected_load_values[col]
            want_addresses = [stream["load_addrs"][col] + 4 * i
                              for i in range(len(want_values))]
            if ([value for _, value in got_loads] != want_values or
                    [address for address, _ in got_loads] != want_addresses):
                load_diffs[col] = {"got_count": len(got_loads), "want_count": len(want_values),
                                   "got_head": got_loads[:10],
                                   "want_values_head": want_values[:10],
                                   "want_addresses_head": want_addresses[:10]}
            got_stores = [address for address, _ in direct["stores"][col]]
            want_stores = [address for address, _ in stream["store_map"][col]]
            if got_stores != want_stores:
                store_diffs[col] = {"got_count": len(got_stores), "want_count": len(want_stores),
                                    "got_head": got_stores[:10], "want_head": want_stores[:10]}
        raise AssertionError(f"Direct memory stream mismatch: {benchmark} N={n} {mode}; "
                             f"loads={load_diffs}, stores={store_diffs}")
    if not correct:
        mismatch = [(i, got, want) for i, (got, want) in
                    enumerate(zip(result, stream["expected"])) if got != want][:8]
        raise AssertionError(f"{benchmark} N={n} {mode} result mismatch: {mismatch}")

    inverse_pc_map = {new: old for old, new in pc_map.items()}
    moved_original_pcs = {branch.pc for group in relocated.values() for branch in group}
    original_branch_visits = {
        str(inverse_pc_map.get(pc, pc) if pc not in moved_original_pcs else pc): count
        for pc, count in sorted(sim.branch_visits.items())
    }
    original_branch_taken = {
        str(inverse_pc_map.get(pc, pc) if pc not in moved_original_pcs else pc): count
        for pc, count in sorted(sim.branch_taken.items())
    }
    pe_config = len(sim.instrs) * cgra.N_COLS
    lcu_config = lcu_branches if lcu_enabled else 0
    trace_payload = repr((direct["loads"], direct["stores"])).encode("ascii")
    return {
        "benchmark": benchmark, "size": n, "architecture": mode,
        "instruction_file": str(instruction_path.relative_to(ROOT)),
        "pcs_instructions": len(sim.instrs), "execution_cycles": sim.aproxcycles,
        "pe_configuration_cycles": pe_config, "lcu_configuration_cycles": lcu_config,
        "configuration_cycles": pe_config + lcu_config,
        "total_cycles": sim.aproxcycles + pe_config + lcu_config,
        "correct": correct, "written": sum(value is not None for value in result),
        "total_outputs": n * n, "pcs_executed": sim.cycles,
        "control_pcs_static": sorted(map(int, original_branch_visits)),
        "control_branch_events": sum(sim.branch_visits.values()),
        "control_taken_branches": sum(sim.branch_taken.values()),
        "branch_events_by_pc": {str(pc): count for pc, count in sorted(sim.branch_visits.items())},
        "taken_branches_by_pc": {str(pc): count for pc, count in sorted(sim.branch_taken.items())},
        "branch_events_by_original_simple_pc": original_branch_visits,
        "taken_branches_by_original_simple_pc": original_branch_taken,
        "lcu_units": len(sim.lcus), "lcu_operations": sim.lcu_operation_count,
        "lcu_branch_operations": sim.lcu_branch_count,
        "lcu_issue_cycles": sim.lcu_issue_cycles,
        "lcu_sync_stalls": sim.lcu_sync_stalls,
        "removed_control_pcs": removed_pcs,
        "removed_control_pc_dynamic_visits": sum(
            sim.branch_visits[branch.pc]
            for group in relocated.values() for branch in group),
        "direct_load_accesses": sum(map(len, direct["loads"])),
        "direct_store_accesses": sum(map(len, direct["stores"])),
        "direct_load_trace_validated": bool(load_ok),
        "direct_store_trace_validated": bool(store_ok),
        "direct_pointer_endpoints_validated": bool(load_pointer_ok and store_pointer_ok),
        "direct_access_trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
        "control_trace_sha256": sim.control_digest.hexdigest(),
    }


def run_indirect_lcu_case(benchmark, n):
    """Run the existing 4x4 baseline and LCU-only configurations."""
    baseline = lcu_ref.run_case(benchmark, n, "OE_4x4", SEED, trace=False)
    global_lcu = lcu_ref.run_case(benchmark, n, "OE_4x4_1LCU", SEED, trace=False)
    if not baseline["correct"] or not global_lcu["correct"]:
        raise AssertionError(f"Indirect baseline/LCU result mismatch at {benchmark} N={n}")
    for key in ("control_trace_sha256", "pc_visits", "dynamic_pe_operations",
                "execution_cycles", "written", "total_outputs"):
        if baseline[key] != global_lcu[key]:
            raise AssertionError(f"Indirect baseline/LCU {key} mismatch at {benchmark} N={n}")
    out = []
    for key, source_result, mode in (
        ("OE Baseline", baseline, "OE_4x4"),
        ("OE + LCU global", global_lcu, "OE_4x4_1LCU"),
    ):
        instruction = lcu_ref.BENCHMARK_FILES[benchmark]
        out.append({
            "benchmark": benchmark, "size": n, "architecture": key,
            "instruction_file": str(instruction.relative_to(ROOT)),
            "pcs_instructions": len(lcu_ref.read_kernel(benchmark)) // 5,
            "execution_cycles": source_result["execution_cycles"],
            "pe_configuration_cycles": source_result["pe_configuration_cycles"],
            "lcu_configuration_cycles": source_result["lcu_configuration_cycles"],
            "configuration_cycles": source_result["configuration_cycles"],
            "total_cycles": source_result["total_cycles"],
            "correct": source_result["correct"], "written": source_result["written"],
            "total_outputs": source_result["total_outputs"],
            "pcs_executed": source_result["pcs_executed"],
            "control_pcs_static": source_result["control_pcs_static"],
            "control_branch_events": source_result["control_branch_events"],
            "control_taken_branches": source_result["control_taken_branches"],
            "branch_events_by_pc": source_result["branch_events_by_pc"],
            "taken_branches_by_pc": source_result["taken_branches_by_pc"],
            "branch_events_by_original_simple_pc": None,
            "taken_branches_by_original_simple_pc": None,
            "lcu_units": source_result["lcu_units"],
            "lcu_operations": source_result["lcu_operations"],
            "lcu_branch_operations": source_result["lcu_branch_operations"],
            "lcu_issue_cycles": source_result["lcu_issue_cycles"],
            "lcu_sync_stalls": source_result["lcu_sync_stalls"],
            "removed_control_pcs": [], "removed_control_pc_dynamic_visits": 0,
            "direct_load_accesses": None, "direct_store_accesses": None,
            "direct_load_trace_validated": None,
            "direct_store_trace_validated": None,
            "direct_pointer_endpoints_validated": None,
            "direct_access_trace_sha256": None,
            "control_trace_sha256": source_result["control_trace_sha256"],
        })
    return out


def run_all(sizes=SIZES, output_dir=HERE):
    results = []
    trace_validation = []
    for benchmark in ("MMUL", "GEM"):
        for n in sizes:
            print(f"Running {benchmark} N={n}", flush=True)
            indirect = run_indirect_lcu_case(benchmark, n)
            a, b, c0 = matrices(n)
            stream = prepare_simple_stream(benchmark, n, a, b, c0)
            direct_rows = []
            for label, mode in (
                ("OE Simple Address", "OE_4x4"),
                ("OE Simple Address + LCU global", "OE_4x4_1LCU"),
                ("OE Simple Address + 4 LCUs", "OE_4x5_4LCU"),
            ):
                # Each run receives an independent memory image and fresh
                # shared direct-address pointers.
                run_stream = dict(stream)
                run_stream["memory"] = [row[:] for row in stream["memory"]]
                row = run_simple_case(benchmark, n, mode, run_stream)
                row["architecture"] = label
                direct_rows.append(row)
                print(f"  PASS {label}: {row['total_cycles']} total cycles; "
                      f"direct trace validated={row['direct_load_trace_validated'] and row['direct_store_trace_validated']}",
                      flush=True)
            # Normalize architecture names for report/plot, retaining unit mode.
            indirect[0]["architecture"] = "OE Baseline"
            indirect[1]["architecture"] = "OE + LCU global"
            rows = [indirect[0], direct_rows[0], indirect[1], direct_rows[1], direct_rows[2]]
            input_hash = hashlib.sha256(repr((a, b, c0)).encode("ascii")).hexdigest()
            for row in rows:
                row["configuration"] = row["architecture"]
                row["seed"] = SEED
                row["input_sha256"] = input_hash
            results.extend(rows)
            combined_trace_hashes = {row["direct_access_trace_sha256"] for row in direct_rows}
            control_counts = {(row["control_branch_events"], row["control_taken_branches"])
                              for row in direct_rows}
            if len(control_counts) != 1:
                raise AssertionError(f"LCU changed loop branch outcomes for {benchmark} N={n}: "
                                     f"{sorted(control_counts)}")
            visits_by_pc = {json.dumps(row["branch_events_by_original_simple_pc"], sort_keys=True)
                            for row in direct_rows}
            taken_by_pc = {json.dumps(row["taken_branches_by_original_simple_pc"], sort_keys=True)
                           for row in direct_rows}
            if len(visits_by_pc) != 1 or len(taken_by_pc) != 1:
                raise AssertionError(f"LCU changed per-loop branch outcomes for {benchmark} N={n}")
            trace_validation.append({
                "benchmark": benchmark, "size": n,
                "simple_direct_streams_match_across_lcu_modes": len(combined_trace_hashes) == 1,
                "direct_load_sequence_matches_original_indirect_trace": all(
                    row["direct_load_trace_validated"] is True for row in direct_rows),
                "direct_store_sequence_matches_original_indirect_trace": all(
                    row["direct_store_trace_validated"] is True for row in direct_rows),
                "direct_pointer_endpoints_validated": all(
                    row["direct_pointer_endpoints_validated"] is True for row in direct_rows),
                "loop_branch_counts_match_across_simple_modes": len(control_counts) == 1,
                "loop_branch_counts_match_per_original_pc": len(visits_by_pc) == 1 and len(taken_by_pc) == 1,
                "control_branch_events": direct_rows[0]["control_branch_events"],
                "control_taken_branches": direct_rows[0]["control_taken_branches"],
                "simple_program_control_trace_hashes": {
                    row["configuration"]: row["control_trace_sha256"] for row in direct_rows},
                "direct_access_trace_sha256": sorted(combined_trace_hashes),
            })
            if len(combined_trace_hashes) != 1:
                raise AssertionError(f"LCU changed direct access order for {benchmark} N={n}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "direct_trace_validation.json").write_text(
        json.dumps(trace_validation, indent=2) + "\n")
    fields = list(results[0])
    with (output_dir / "combined_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    print(f"Wrote {len(results)} configurations to {output_dir / 'combined_results.csv'}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    run_all(tuple(args.sizes), args.output_dir)


if __name__ == "__main__":
    main()
