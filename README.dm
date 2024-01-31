# Optimization Algorithms ReadMe

This repository contains implementations of optimization algorithms, including the Hungarian Algorithm and the Simplex Algorithm for linear programming problems. Below is an overview of the provided files and their functionalities.

## Files:

### 1. `Hungarian_method.py`

This Python script provides an implementation of the Hungarian Algorithm for solving the assignment problem. The assignment problem involves finding the optimal assignment of tasks to workers, minimizing the total cost.

#### Functions:

- `reduce_matrix_costs(cost_matrix)`: Reduces the cost matrix by subtracting the minimum value in each row and column.
- `find_min_zeros_row(is_zero_matrix, marked_zero)`: Finds the row with the minimum number of zeros and adds the first zero in that row to the list of marked zeros.
- ...

#### Example Usage:
