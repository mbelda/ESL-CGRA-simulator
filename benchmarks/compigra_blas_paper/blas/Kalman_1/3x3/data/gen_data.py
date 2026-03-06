#!/usr/bin/env python3

import numpy as np
from pathlib import Path


# -------------------------------------------------
# Reference computation (same as C code)
# -------------------------------------------------
def kalman_reference(A, x, P, NI):

    Ax = np.zeros(NI, dtype=np.int32)
    AP = np.zeros(NI * NI, dtype=np.int32)

    # Ax = A * x
    for i in range(NI):
        s = 0
        for k in range(NI):
            s += int(A[i*NI + k]) * int(x[k])
        Ax[i] = s

    # AP = A * P
    for i in range(NI):
        for j in range(NI):
            s = 0
            for k in range(NI):
                s += int(A[i*NI + k]) * int(P[k*NI + j])
            AP[i*NI + j] = s

    return Ax, AP


# -------------------------------------------------
# Generate + write
# -------------------------------------------------
def generate_and_write(NI, seed=None):

    if seed is not None:
        np.random.seed(seed)

    NI = int(NI)

    data_dir = Path(".")
    size_tag = f"{NI}"

    h_filename = data_dir / f"data_{size_tag}.h"
    npz_filename = data_dir / f"data_{size_tag}.npz"
    latest_npz = data_dir / "data.npz"

    # -------------------------------------------------
    # Generate 1D data
    # -------------------------------------------------
    A = np.random.randint(-5, 6, size=(NI * NI), dtype=np.int32)
    x = np.random.randint(-5, 6, size=(NI), dtype=np.int32)
    P = np.random.randint(-5, 6, size=(NI * NI), dtype=np.int32)

    # Golden computation
    Ax, AP = kalman_reference(A, x, P, NI)

    # -------------------------------------------------
    # Write C header
    # -------------------------------------------------
    def write_array(f, name, arr):
        f.write(f"int {name}[{len(arr)}] = {{\n    ")
        f.write(", ".join(str(int(v)) for v in arr))
        f.write("\n};\n\n")

    with open(h_filename, "w") as f:

        f.write("#ifndef DATA_H\n")
        f.write("#define DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"#define NI {NI}\n\n")

        write_array(f, "A", A)
        write_array(f, "x", x)
        write_array(f, "P", P)

        write_array(f, "Ax_expected", Ax)
        write_array(f, "AP_expected", AP)

        f.write("#endif\n")

    # -------------------------------------------------
    # Write NPZ
    # -------------------------------------------------
    np.savez(
        npz_filename,
        A=A,
        x=x,
        P=P,
        Ax_expected=Ax,
        AP_expected=AP,
        NI=NI
    )

    np.savez(
        latest_npz,
        A=A,
        x=x,
        P=P,
        Ax_expected=Ax,
        AP_expected=AP,
        NI=NI
    )

    print("Generated:")
    print(h_filename)
    print(npz_filename)
    print(latest_npz)


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():

    import argparse

    parser = argparse.ArgumentParser(description="Generate Kalman test data")

    parser.add_argument("--NI", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()

    generate_and_write(
        NI=args.NI,
        seed=args.seed
    )


if __name__ == "__main__":
    main()