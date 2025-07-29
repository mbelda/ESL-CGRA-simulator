import math
import csv
import sys
import os

## Convert the output of SAT-MapIt into a csv file compatible with the simulator
def convert(infile, outfile, version=""):

    outfile = outfile.split(".")[0] + "{}." + outfile.split(".")[-1]
    outfile = outfile.format(version)

    # Read the input file (output of SAT-MapIt)
    with open(infile, "r") as f:
        lines = f.readlines()

    reading_conf = False
    last_time_stamp = 0
    current_time_stamp = 0
    configuration = []

    conf_set = []

    for line in lines:

        # Start reading instructions
        if line.startswith("T ="):
            reading_conf = True
            configuration = []
            current_time_stamp = int(line.split(" ")[-1])

            # stop reading, reached the last configuration
            if last_time_stamp > current_time_stamp:
                reading_conf = False
                break
            else:
                last_time_stamp = current_time_stamp
                conf_set.append(configuration)

        if reading_conf:
            configuration.append(line)

    if not conf_set:
        print("No configurations found.")
        return

    # counts the nodes and infer rows and columns (always assumes a squared mesh)
    n_nodes = len(conf_set[0][1:])
    n_cols = int(math.sqrt(n_nodes))
    n_rows = n_cols

    # Write the output file
    with open(outfile, "w", newline='') as f:
        writer = csv.writer(f)

        for conf in conf_set:
            time = conf[0]                          # Line with the timestamp
            time = int(time.split(" ")[-1].strip()) # extract timestamp

            instrs = conf[1:]   # Set of all instructions in the current configuration

            # Write the timestamp
            writer.writerow([time])

            rows = [[instrs[(n_cols * r) + c].strip() for c in range(n_cols)] for r in range(n_rows)]

            for r in rows:
                writer.writerow(r)


def main():
    if len(sys.argv) < 3:
        print("Uso: python script.py <archivo_entrada> <archivo_salida> [version]")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    version = sys.argv[3] if len(sys.argv) > 3 else ""

    if not os.path.exists(infile):
        print(f"Error: el archivo '{infile}' no existe.")
        sys.exit(1)

    convert(infile, outfile, version)
    print(f"Archivo convertido guardado en '{outfile.split('.')[0] + version + '.' + outfile.split('.')[-1]}'")


if __name__ == "__main__":
    main()
