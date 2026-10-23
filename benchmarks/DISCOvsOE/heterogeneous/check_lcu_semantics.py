"""Focused checks for nested software loops and synchronized row LCUs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgra import ARCHITECTURES, HeterogeneousCGRA, HeterogeneousPC


def empty_pc(rows, instruction):
    return HeterogeneousPC([["NOP"] * 4 for _ in range(rows)], [instruction] * rows)


def check_nested_loops():
    rows = 2
    program = [
        empty_pc(rows, "SADD R0, ZERO, ZERO"),   # outer counter = 0
        empty_pc(rows, "SADD R1, ZERO, 2"),      # outer bound
        empty_pc(rows, "SADD R2, ZERO, ZERO"),   # inner counter = 0
        empty_pc(rows, "SADD R3, ZERO, 3"),      # inner bound
        empty_pc(rows, "SADD R2, R2, 1"),        # inner body/counter update
        empty_pc(rows, "BNE R2, R3, 4"),         # inner loop
        empty_pc(rows, "SADD R0, R0, 1"),         # outer counter update
        empty_pc(rows, "BNE R0, R1, 2"),         # restart inner loop
    ]
    sim = HeterogeneousCGRA(ARCHITECTURES["OE_2x5_LCU"], program)
    stats = sim.run()
    assert [lcu.regs["R0"] for lcu in sim.lcus] == [2, 2]
    assert [lcu.regs["R2"] for lcu in sim.lcus] == [3, 3]
    assert stats["lcu_taken_branches"] == 10
    return stats


def check_divergence_is_rejected():
    rows = 2
    program = [
        HeterogeneousPC([["NOP"] * 4 for _ in range(rows)],
                        ["SADD R0, ZERO, 0", "SADD R0, ZERO, 1"]),
        HeterogeneousPC([["NOP"] * 4 for _ in range(rows)],
                        ["BNE R0, ZERO, 0", "BNE R0, ZERO, 0"]),
    ]
    sim = HeterogeneousCGRA(ARCHITECTURES["OE_2x5_LCU"], program)
    try:
        sim.run()
    except RuntimeError as exc:
        assert "divergence" in str(exc)
        return str(exc)
    raise AssertionError("Different row-LCU decisions must be rejected")


if __name__ == "__main__":
    nested = check_nested_loops()
    divergence = check_divergence_is_rejected()
    print("nested loops: PASS", nested)
    print("LCU synchronization: PASS", divergence)
