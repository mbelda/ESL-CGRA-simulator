import numpy as np

# Define matrices A and B
A = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12,13,14,15],
    [16,17,18,19,20],
    [21, 22, 23, 24,25]
])

B = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
    [130, 140, 150, 160],
    [170, 180, 190, 200]
])

# Perform matrix multiplication
C = np.dot(A, B)

# Print the result
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nResultant Matrix C:")
print(C)
