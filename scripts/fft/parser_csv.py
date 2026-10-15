import argparse
import csv
import re
import sys


def process_csv(input_file, output_file):
    # Regular expression to catch "Cycle/Instruction Step X" and extract X
    step_regex = re.compile(r"Cycle/Instruction\s+Step\s+(\d+)", re.IGNORECASE)

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

                row_str = " ".join(row).strip()

                # Skip top metadata headers
                if row_str.startswith("#"):
                    continue

                # Check if the row indicates a new Cycle/Instruction Step
                match = step_regex.search(row_str)
                if match:
                    step_num = match.group(1)
                    writer.writerow([step_num, "", "", ""])
                else:
                    # Clean whitespace and pad/slice to ensure exactly 4 columns
                    cleaned_row = [col.strip() for col in row]
                    while len(cleaned_row) < 4:
                        cleaned_row.append("")
                    writer.writerow(cleaned_row[:4])

        print(f"Success: Processed '{input_file}' -> '{output_file}'")

    except FileNotFoundError:
        print(
            f"Error: The input file '{input_file}' was not found.",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Format Kernel Execution Layout CSV files."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input CSV file (e.g., input.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save the formatted output CSV file (e.g., output.csv)",
    )

    args = parser.parse_args()

    # Run the main processing function with the arguments
    process_csv(args.input, args.output)