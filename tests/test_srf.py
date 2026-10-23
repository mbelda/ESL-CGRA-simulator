import unittest

from cgra import CGRA, N_COLS, N_ROWS


def make_kernel(*cycles):
    kernel = []
    for number, operations in enumerate(cycles):
        kernel.append([str(number)] + [""] * (N_COLS - 1))
        for row in range(N_ROWS):
            kernel.append(operations[row])
    return kernel


def empty_cycle():
    return [["NOP"] * N_COLS for _ in range(N_ROWS)]


class ScalarRegisterFileTests(unittest.TestCase):
    def make_cgra(self, *cycles):
        return CGRA(make_kernel(*cycles), [], None, None)

    def test_one_write_and_one_read_share_cycle_snapshot(self):
        operations = empty_cycle()
        operations[0][0] = "SADD SRF0, R0, R1"
        operations[1][0] = "SADD R0, SRF0, R1"
        next_cycle = empty_cycle()
        next_cycle[2][0] = "SADD R2, SRF0, 0"
        cgra = self.make_cgra(operations, next_cycle)
        cgra.srf["SRF0"] = 5
        cgra.cells[0][0].regs["R0"] = 7
        cgra.cells[0][0].regs["R1"] = 3
        cgra.cells[1][0].regs["R1"] = 2

        cgra.step()

        self.assertEqual(cgra.cells[1][0].regs["R0"], 7)
        self.assertEqual(cgra.srf["SRF0"], 10)

        cgra.step()
        self.assertEqual(cgra.cells[2][0].regs["R2"], 10)

    def test_multiple_readers_of_same_register_are_allowed(self):
        operations = empty_cycle()
        operations[0][0] = "SADD R0, SRF2, 1"
        operations[1][0] = "SADD R0, SRF2, 2"
        cgra = self.make_cgra(operations, empty_cycle())
        cgra.srf["SRF2"] = 9

        cgra.step()

        self.assertEqual(cgra.cells[0][0].regs["R0"], 10)
        self.assertEqual(cgra.cells[1][0].regs["R0"], 11)

    def test_different_register_accesses_are_rejected_before_execution(self):
        operations = empty_cycle()
        operations[0][0] = "SADD R0, SRF0, 1"
        operations[1][0] = "SADD R0, SRF1, 1"
        cgra = self.make_cgra(operations)
        cgra.cells[0][0].regs["R0"] = 42

        with self.assertRaisesRegex(RuntimeError, r"cycle 0.*SRF0.*SRF1"):
            cgra.step()

        self.assertEqual(cgra.cells[0][0].regs["R0"], 42)

    def test_two_writers_are_rejected_with_row_major_locations(self):
        operations = empty_cycle()
        operations[0][1] = "SADD SRF3, 1, 2"
        operations[1][0] = "SADD SRF3, 3, 4"
        cgra = self.make_cgra(operations)

        with self.assertRaisesRegex(RuntimeError, r"SRF3.*RC\(0,1\).*RC\(1,0\)"):
            cgra.step()

        self.assertEqual(cgra.srf["SRF3"], 0)

    def test_one_instruction_cannot_read_two_srf_registers(self):
        operations = empty_cycle()
        operations[0][0] = "SADD R0, SRF0, SRF1"
        cgra = self.make_cgra(operations)

        with self.assertRaises(RuntimeError):
            cgra.step()

    def test_write_and_read_of_different_registers_are_rejected(self):
        operations = empty_cycle()
        operations[0][0] = "SADD SRF0, 1, 2"
        operations[1][0] = "SADD R0, SRF1, 3"
        cgra = self.make_cgra(operations)

        with self.assertRaises(RuntimeError):
            cgra.step()

    def test_writers_to_different_registers_are_rejected(self):
        operations = empty_cycle()
        operations[0][0] = "SADD SRF0, 1, 2"
        operations[1][0] = "SADD SRF1, 3, 4"
        cgra = self.make_cgra(operations)

        with self.assertRaises(RuntimeError):
            cgra.step()

    def test_srf8_is_rejected(self):
        operations = empty_cycle()
        operations[0][0] = "SADD R0, SRF8, 1"
        cgra = self.make_cgra(operations)

        with self.assertRaisesRegex(ValueError, r"SRF8.*RC\(0,0\)"):
            cgra.step()

    def test_local_register_instruction_keeps_existing_behavior(self):
        operations = empty_cycle()
        operations[0][0] = "SADD R0, R1, R2"
        cgra = self.make_cgra(operations, empty_cycle())
        cgra.cells[0][0].regs["R1"] = 4
        cgra.cells[0][0].regs["R2"] = 6

        cgra.step()

        self.assertEqual(cgra.cells[0][0].regs["R0"], 10)
        self.assertEqual(cgra.srf, {f"SRF{i}": 0 for i in range(8)})


if __name__ == "__main__":
    unittest.main()
