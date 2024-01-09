import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=DeprecationWarning)
np.set_printoptions(suppress=True)

# Implementation of the Simplex algorithm for maximization problems

# Maximize: 10x + 5y 
# Constraints:
#  4x + 2y <= 900
#  2x + 4y <= 1000
#   x +  y   <= 300
#  x, y >= 0

#c = np.array([6, 14, 13, 0, 0])
#A = np.matrix([[1, 2, 4, 1 , 0], [0.5, 2, 1, 0, 1]])
#b = np.array([[900], [1000]])

c = np.array([2, 1.5, 0, 0])
A = np.matrix([[6, 3, 1 , 0], [75, 100, 0, 1]])
b = np.array([[1200], [25000]])


def pivot(table, nonbase, nonbase_idx):
    print("\n**************************\n")
    print(np.round(table, 2))
    omega = table[0, 1:-2]
    below_omega = table[1:, 1:-2]
    print("nonbase:", nonbase, "  nonbase_idx:", nonbase_idx)
    j_candidates_A = []
    j_candidates_B = []
    j_candidates_C = []
    for (index, val) in enumerate(nonbase_idx):
        print("val:", val)
        aj = A[:, val]
        print("aj:\n", aj)
        print("nonbase[index]:", nonbase[index])
        cj = omega.dot(aj) - nonbase[index]
        print("cj:", cj)
        j_candidates_A.append(cj)
        j_candidates_B.append(aj)
        j_candidates_C.append(val)
    # Termination condition
    flag = 0
    for candidate in j_candidates_A:
        if candidate < 0:
            flag = 1
    if flag == 0:
        return (False, False)
    min_j_candidate_A = j_candidates_A[0]
    min_j_candidate_index = 0
    for (index, candidate) in enumerate(j_candidates_A):
        if min_j_candidate_A > candidate:
            min_j_candidate_A = candidate
            min_j_candidate_index = index
    min_j_candidate_value = min_j_candidate_A
    min_j_candidate_array = j_candidates_B[min_j_candidate_index]
    min_j_candidate_coeffs_index = j_candidates_C[min_j_candidate_index]
    print("min_j_candidate_value:", min_j_candidate_value)
    print("min_j_candidate_array:", min_j_candidate_array)
    print("min_j_candidate_coeffs_index:", min_j_candidate_coeffs_index)
    # Insert pivot column
    table[0, -1] = min_j_candidate_value
    # Cast to list because table[1:, -1] has shape (2,) and this is (2,1)
    table[1:, -1] = list(below_omega.dot(min_j_candidate_array))
    # Determine pivot element and pivot row
    min_elements = []
    for index in range(A.shape[0]):
        divisor = table[1:, -2]
        divider = table[1:, -1]
        division = divisor[index] / divider[index]
        min_elements.append((division, index + 1))
    print("min_elements:", min_elements)
    (min_element, pivot_row) = min(min_elements)
    print("min_element & pivot_row :", min_element, pivot_row)
    pivot_elem = table[pivot_row, -1]
    print("pivot_elem =", pivot_elem)
    # Update the first column
    table[pivot_row, 0] = min_j_candidate_coeffs_index
    print("Updated first column with", min_j_candidate_coeffs_index)
    print(np.round(table, 2))
    return (pivot_row, pivot_elem)


def reset(table):
    print("\n----------------------------------------------\n")
    print("c:", c)
    base = []
    base_idx = []
    nonbase = []
    nonbase_idx = []
    for (index, val) in enumerate(c):
        if index in table[1:, 0]:
            base.append(val)
            base_idx.append(index)
        else:
            nonbase.append(val)
            nonbase_idx.append(index)
    print("base:", nonbase, "     base_idx:", nonbase_idx)
    print("nonbase:", nonbase, "  nonbase_idx:", nonbase_idx)
    return (
        base,
        base_idx,
        nonbase,
        nonbase_idx,
    )


def update_table(table, pivot_row, pivot_elem):
    print("Calling update_table...")
    # Update all elements except the pivot row, it should not affect the others
    for row in range(A.shape[0] + 1):
        for col in range(1, A.shape[0] + 2):
            if row == pivot_row:
                continue
            approp_row = table[pivot_row, col]
            approp_col = table[row, -1]
            table[row, col] = table[row, col] - (approp_row * approp_col / pivot_elem)
    # Update the pivot row
    for row in range(A.shape[0] + 1):
        for col in range(1, A.shape[0] + 2):
            if row == pivot_row:
                table[row, col] = table[row, col] / pivot_elem


def simplex_maximize(c, A, b):
    # Initial filling of the table
    table = np.zeros((A.shape[0] + 1, A.shape[0] + 3))
    print("Initial table:\n")
    print(table,"\n")
    # Fill the first column
    base_idx = []
    for (index, val) in enumerate(c):
        if val == 0:
            base_idx.append(index)
    table[1:, 0] = base_idx
    base = np.zeros((1, len(base_idx)))
    for (index, indexValue) in enumerate(base_idx):
        base[0, index] = c[indexValue]        
    print("base_idx:", base_idx, " base:", base)

    B_inv = np.linalg.inv(A[:, base_idx[0] :])
    print("\nB inv: ")
    print(B_inv)
    table[1:, 1:-2] = B_inv

    b = B_inv.dot(b)
    print("b:")
    print(b)
    # Cast to list because table[1:, -2] has shape (2,) and this is (2,1)
    table[1:, -2] = list(b)

    omega = base.dot(B_inv)
    print("omega:", omega)
    table[0, 1:-2] = omega[0, :]
    Cbb = base.dot(b)
    print("Cbb: ")
    print(Cbb)
    table[0, -2] = Cbb

    print("\nFinished inserting!\n")
    print(np.round(table, 2))

    # Reset and get nonbase and nonbase_idx, they are created here
    (
        base,
        base_idx,
        nonbase,
        nonbase_idx,
    ) = reset(table)

    while True:
        # Calculation of the pivot column, pivot element, and update of the first column
        (pivot_row, pivot_elem) = pivot(table, nonbase, nonbase_idx)
        if pivot_row == False:
            print(np.round(table, 2))
            print("Simplex algorithm finished!")
            print("The result is: ", table[0, -2])
            for i in range(table.shape[0] - 1):
                print(
                    "Base variable",
                    int(table[i + 1, 0]),
                    "has coefficient",
                    table[i + 1, -2],
                )
            return
        (
            base,
            base_idx,
            nonbase,
            nonbase_idx,
        ) = reset(table)
        # Update all elements except the first and last column
        # Calculate based on pivot_row, pivot_column (from the table directly) and pivot_elem
        update_table(table, pivot_row, pivot_elem)
        print(np.round(table, 2))


simplex_maximize(c, A, b)
