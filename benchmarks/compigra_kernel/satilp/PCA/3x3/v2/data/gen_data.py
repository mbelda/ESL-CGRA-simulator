#!/usr/bin/env python3

import numpy as np
from pathlib import Path


# -------------------------------------------------
# Exact same PCA as in C (integer version)
# -------------------------------------------------
def pca_reference(inputX, mu, NI, NJ):

    Xc = np.zeros((NI, NJ), dtype=np.int32)
    XcT = np.zeros((NJ, NI), dtype=np.int32)
    C = np.zeros((NJ, NJ), dtype=np.int32)

    # center
    for i in range(NI):
        for j in range(NJ):
            Xc[i][j] = np.int32(inputX[i][j] - mu[j])
            XcT[j][i] = Xc[i][j]

    # C = XcT * Xc
    for i in range(NJ):
        for j in range(NJ):
            s = 0
            for k in range(NI):
                s += XcT[i][k] * Xc[k][j]
            C[i][j] = s
    
    # Print C as matrix
    print("C:")
    for i in range(NJ):
        print("  ", end="")
        for j in range(NJ):
            print(f"{C[i][j]:6d} ", end="")
        print()
    # Print inputX as matrix
    print("inputX:")
    for i in range(NI):
        print("  ", end="")
        for j in range(NJ):
            print(f"{inputX[i][j]:6d} ", end="")
        print()

    return Xc.copy(), XcT.copy(), C.copy()

# -------------------------------------------------
# Generate + write
# -------------------------------------------------
def generate_and_write(NI, NJ, seed=None):

    if seed is not None:
        np.random.seed(seed)

    NI = int(NI)
    NJ = int(NJ)

    data_dir = Path(".")

    size_tag = f"{NI}x{NJ}"

    h_filename = data_dir / f"data_{size_tag}.h"
    npz_filename = data_dir / f"data_{size_tag}.npz"
    latest_npz = data_dir / "data.npz"

    # -------------------------------------------------
    # Generate integer data (small range to avoid overflow)
    # -------------------------------------------------
    inputX_2d = np.random.randint(-5, 6, size=(NI, NJ), dtype=np.int32)
    #inputX_2d = np.array([np.ones(NJ, dtype=np.int32) if i % 2 == 0 
    #                  else np.zeros(NJ, dtype=np.int32) 
    #                  for i in range(NI)])
    inputX_2d = np.full((NI, NJ), -2, dtype=np.int32)
    inputX_2d[:, ::2] = 1
    #mu = np.random.randint(-5, 6, size=(NJ,), dtype=np.int32)
    #inputX_2d = np.ones((NI, NJ), dtype=np.int32)
    mu = np.ones((NJ,), dtype=np.int32)

    # Golden computation
    Xc_2d, XcT_2d, C_2d = pca_reference(inputX_2d, mu, NI, NJ)

    # Flatten row-major
    inputX = inputX_2d.flatten()
    Xc = Xc_2d.flatten()
    XcT = XcT_2d.flatten()
    C = C_2d.flatten()

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

        f.write(f"#define NI {NI}\n")
        f.write(f"#define NJ {NJ}\n\n")

        write_array(f, "inputX", inputX)
        write_array(f, "mu", mu)
        #write_array(f, "inputX", [1 for _ in range(len(inputX))])  # dummy input
        #write_array(f, "mu", [1 for _ in range(len(mu))])  # dummy mean
        write_array(f, "Xc_golden", Xc)
        write_array(f, "XcT_golden", XcT)
        write_array(f, "C_golden", [0 for _ in range(len(C))])  # dummy C, to be filled by the kernel
        write_array(f, "expected_C", C)

        f.write("#endif\n")

    # -------------------------------------------------
    # Write NPZ
    # -------------------------------------------------
    np.savez(
        npz_filename,
        inputX=inputX,
        mu=mu,
        Xc_golden=Xc,
        XcT_golden=XcT,
        C_golden=C,
        expected_C=C,
        NI=NI,
        NJ=NJ
    )

    np.savez(
        latest_npz,
        inputX=inputX,
        mu=mu,
        Xc_golden=Xc,
        XcT_golden=XcT,
        C_golden=C,
        expected_C=C,
        NI=NI,
        NJ=NJ
    )

    print(f"Generated:")
    print(f"  {h_filename}")
    print(f"  {npz_filename}")
    print(f"  {latest_npz}")

    return inputX, mu, Xc, XcT, C


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate PCA integer data")

    parser.add_argument("--NI", type=int, required=True)
    parser.add_argument("--NJ", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()

    generate_and_write(
        NI=args.NI,
        NJ=args.NJ,
        seed=args.seed
    )


if __name__ == "__main__":
    main()