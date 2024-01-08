import numpy as np

def reduce_matrix_costs(cost_matrix):
    print("Initial matrix:")
    print(cost_matrix)

    # Subtracting the minimum in each row
    reduced_matrix = cost_matrix - np.min(cost_matrix, axis=1)[:, np.newaxis]

    print("\nAfter subtracting the minimum in each row:")
    print(reduced_matrix)

    # Subtracting the minimum in each column where there is no zero (only in columns without zero)
    for j in range(reduced_matrix.shape[1]):
        column_values = reduced_matrix[:, j]
        non_zero_indices = np.nonzero(column_values)[0]

        if len(non_zero_indices) > 0 and np.count_nonzero(reduced_matrix[:, j] == 0) == 0:
            min_val = np.min(column_values[non_zero_indices])
            reduced_matrix[:, j] -= min_val

    print("\nAfter subtracting the minimum in the selected column:")
    print(reduced_matrix)

    return reduced_matrix

def find_min_zeros_row(is_zero_matrix, marked_zero):
    # Finds the row with the minimum number of zeros and adds the first zero in that row to the list of marked zeros.
    min_row = [float('inf'), -1]

    for row in range(is_zero_matrix.shape[0]):
        if np.sum(is_zero_matrix[row]) > 0 and min_row[0] > np.sum(is_zero_matrix[row]):
            min_row = [np.sum(is_zero_matrix[row]), row]

    zero_index = np.where(is_zero_matrix[min_row[1]])[0][0]
    marked_zero.append((min_row[1], zero_index))
    is_zero_matrix[min_row[1]] = False
    is_zero_matrix[:, zero_index] = False

def mark_matrix(cost_matrix):
    transformed_matrix = reduce_matrix_costs(cost_matrix)
    is_zero_matrix = (transformed_matrix == 0)  # If there is 0 in the cost matrix, the corresponding index is True
    is_zero_matrix_copy = is_zero_matrix.copy()

    marked_zeros_idx = []
    while True in is_zero_matrix_copy:
        find_min_zeros_row(is_zero_matrix_copy, marked_zeros_idx)

    marked_zero_rows = [row for row, _ in marked_zeros_idx]

    unmarked_rows = list(set(range(transformed_matrix.shape[0])) - set(marked_zero_rows))

    marked_cols = []
    flag = True
    while flag:
        flag = False
        for i in range(len(unmarked_rows)):
            row_array = is_zero_matrix[unmarked_rows[i], :]

            for j in range(row_array.shape[0]):
                if row_array[j] and j not in marked_cols:
                    marked_cols.append(j)
                    flag = True

        for row_num, col_num in marked_zeros_idx:
            if row_num not in unmarked_rows and col_num in marked_cols:
                unmarked_rows.append(row_num)
                flag = True

    marked_rows = list(set(range(cost_matrix.shape[0])) - set(unmarked_rows))

    return marked_zeros_idx, marked_rows, marked_cols

def update_matrix(cost_matrix, cover_rows, cover_cols):
    current_matrix = cost_matrix
    non_zero_elements = []

    for row_index in range(len(current_matrix)):
        if row_index not in cover_rows:
            for col_index in range(len(current_matrix[row_index])):
                if col_index not in cover_cols:
                    non_zero_elements.append(current_matrix[row_index, col_index])

    min_non_zero = min(non_zero_elements)

    for row_index in range(len(current_matrix)):
        if row_index not in cover_rows:
            for col_index in range(len(current_matrix[row_index])):
                if col_index not in cover_cols:
                    current_matrix[row_index, col_index] -= min_non_zero

    for row_index in range(len(cover_rows)):
        for col_index in range(len(cover_cols)):
            current_matrix[cover_rows[row_index], cover_cols[col_index]] += min_non_zero

    return current_matrix

def hungarian_algorithm(cost_matrix):
    # Adding and subtracting the minimum element from the corresponding places
    dim = cost_matrix.shape[0]
    cur_mat = cost_matrix

    zero_count = 0
    while zero_count < dim:
        zero_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = update_matrix(cur_mat, marked_rows, marked_cols)

    return zero_pos

def solve_hungarian(cost_matrix, pos):
    total = 0
    solution_matrix = np.zeros_like(cost_matrix)

    for row, col in pos:
        total += cost_matrix[row, col]
        solution_matrix[row, col] = cost_matrix[row, col]

    return total, solution_matrix

cost_matrix = np.array([[14, 9, 12, 8, 16],
                        [8, 7, 9, 9, 14],
                        [9, 11, 10, 10, 12],
                        [10, 8, 8, 6, 14],
                        [11, 9, 10, 7, 13]])

zero_pos = hungarian_algorithm(cost_matrix.copy())
result, result_matrix = solve_hungarian(cost_matrix, zero_pos)
print(f"\nAssignment Problem Result: {result:.0f}\n{result_matrix}\n")
