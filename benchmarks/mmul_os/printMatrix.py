import csv
import sys

def read_csv_rows(file_path):
    array_list = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')  # Change delimiter if needed


        # Assuming each subsequent row has two values
        for row in csv_reader:
            if len(row) == 2:  # Ensure the row has exactly two values
                # Convert the values to integers and create a list of arrays
                try:
                    array_list.append([int(row[0]), int(row[1])])
                except ValueError:
                    print("Error: Values in CSV file are not integers.")
                    return None

    return array_list

def print_matrix_for_range(array_list, x, y, num_columns):
    rows = ((y-x)/4+1)/num_columns
    j = 0
    for row in array_list:
        if row[0] >= x and row[0]<=y: 
            print(row[1], end=" ")
            j+=1
            if j%num_columns==0:
                print()

# Example usage with command-line arguments
if len(sys.argv) != 5:
    print("Usage: python script.py file_path x_value y_value num_columns")
    sys.exit(1)

file_path = sys.argv[1]
x_value = int(sys.argv[2])
y_value = int(sys.argv[3])
num_columns = int(sys.argv[4])

list_of_arrays = read_csv_rows(file_path)

if list_of_arrays:
    print_matrix_for_range(list_of_arrays, x_value, y_value, num_columns)
else:
    print("Error: CSV file does not have rows with two values or has invalid values.")
