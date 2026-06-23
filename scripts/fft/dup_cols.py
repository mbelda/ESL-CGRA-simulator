import argparse
import csv
import re
import sys


def duplicate_columns(input_file, output_file):
    # Matches the Step lines (e.g., "0,,,") so we don't accidentally overwrite them
    step_pattern = re.compile(r"^\d+$")

    try:
        with open(
            input_file, mode="r", newline="", encoding="utf-8"
        ) as infile, open(
            output_file, mode="w", newline="", encoding="utf-8"
        ) as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

            for row in reader:
                if not row:
                    continue

                # Ensure the row has elements to check
                first_col = row[0].strip()

                if step_pattern.match(first_col):
                    # It's a header step line (like 0,,,), pass it through exactly as is
                    writer.writerow([first_col, "", "", ""])
                else:
                    # It's an instruction row. Take the 1st column operation
                    # and clone it across all 4 columns.
                    duplicated_row = [first_col, first_col, first_col, first_col]
                    writer.writerow(duplicated_row)

        print(f"Success: Duplicated columns '{input_file}' -> '{output_file}'")

    except FileNotFoundError:
        print(
            f"Error: The input file '{input_file}' was not found.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Duplicate column 1 operations across all 4 columns in a step-formatted CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the formatted input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save the duplicated output CSV file",
    )

    args = parser.parse_args()
    duplicate_columns(args.input, args.output)