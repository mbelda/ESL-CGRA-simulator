#!/usr/bin/env python3

import numpy as np
from pathlib import Path


# -------------------------------------------------
# Reference computation (same as C code)
# -------------------------------------------------
def kalman2_reference(A, Q, AP, NI):

    AT = np.zeros(NI * NI, dtype=np.int32)
    APA = np.zeros(NI * NI, dtype=np.int32)
    P = np.zeros(NI * NI, dtype=np.int32)

    # AT = transpose(A)
    for i in range(NI):
        for j in range(NI):
            AT[j*NI + i] = A[i*NI + j]

    # APA = AP * AT
    for i in range(NI):
        for j in range(NI):
            s = 0
            for k in range(NI):
                s += int(AP[i*NI + k]) * int(AT[k*NI + j])
            APA[i*NI + j] = s

    # P = APA + Q
    for i in range(NI):
        for j in range(NI):
            P[i*NI + j] = APA[i*NI + j] + Q[i*NI + j]

    return AT, APA, P


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
    Q = np.random.randint(-5, 6, size=(NI * NI), dtype=np.int32)
    AP = np.random.randint(-5, 6, size=(NI * NI), dtype=np.int32)

    # Golden computation
    AT, APA, P = kalman2_reference(A, Q, AP, NI)

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
        write_array(f, "Q", Q)
        write_array(f, "AP", AP)

        write_array(f, "AT_expected", AT)
        write_array(f, "APA_expected", APA)
        write_array(f, "P_expected", P)

        f.write("#endif\n")

    # -------------------------------------------------
    # Write NPZ
    # -------------------------------------------------
    np.savez(
        npz_filename,
        A=A,
        Q=Q,
        AP=AP,
        AT_expected=AT,
        APA_expected=APA,
        P_expected=P,
        NI=NI
    )

    np.savez(
        latest_npz,
        A=A,
        Q=Q,
        AP=AP,
        AT_expected=AT,
        APA_expected=APA,
        P_expected=P,
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

    parser = argparse.ArgumentParser(description="Generate Kalman_2 test data")

    parser.add_argument("--NI", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()

    generate_and_write(
        NI=args.NI,
        seed=args.seed
    )


if __name__ == "__main__":
    main()