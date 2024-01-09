import numpy as np

def convert_inequality_to_equality(c, A, b):
    num_constraints = len(b)

    A_artificial = np.hstack((A, np.eye(num_constraints)))
    c_combined = np.hstack((c, np.zeros(num_constraints)))
    
    return c_combined, A_artificial

def initialize_simplex(c_combined, A_combined, b):
    num_constraints = len(b)
    num_variables = len(c)

    B_indices = np.arange(num_variables, num_variables + num_constraints)
    
    c = A_combined[:, B_indices]

    while True:
        try:
            B_inv = np.linalg.inv(c)
        except np.linalg.LinAlgError:
            print("The c matrix is not invertible. Choose new basic variables.")
            break

        B_inv_b = np.dot(B_inv, b)
        if np.all(B_inv_b >= 0):
            print("\nBasic variables indices:", B_indices)
            print("\nB matrix:")
            print(c)
            print("\nB inverse:")
            print(B_inv)
            print("\nB_inv * b:")
            print(B_inv_b)
            
            B_values = A_combined[:, B_indices]
            print("\nMatrix c (extracted from A_combined):")
            print(B_values)
            
            return B_indices, B_inv, B_inv_b
        else:
            print("B_inv * b is not non-negative. Choose new basic variables.")
            break

def calculate_W_Cb_B_inv_b(c_combined, B_indices, B_inv, b):
    Cb = c_combined[B_indices]
    W = np.dot(Cb, B_inv)
    Cb_B_inv_b = np.dot(Cb, np.dot(B_inv, b))

    return W, Cb_B_inv_b

def find_pivot_column(W, C, A_combined, B_indices, maximization=True):
    pivot_head_values = []
    non_basic_indices = [i for i in range(len(C)) if i not in B_indices]
    
    print("\nNon basic indices: ", non_basic_indices)
    
    for j in non_basic_indices:
        a_j = A_combined[:, j]
        pivot_head = np.dot(W, a_j) - C[j]
        pivot_head_values.append(pivot_head)

    if maximization:
        pivot_col_index = np.argmin(pivot_head_values)
        pivot_head = np.min(pivot_head_values)
    else:
        pivot_col_index = np.argmax(pivot_head_values)
        pivot_head = np.max(pivot_head_values)

    pivot_col = A_combined[:, non_basic_indices[pivot_col_index]]


    return pivot_col, pivot_head, pivot_col_index, pivot_head_values

def find_pivot_row(B_inv_b, pivot_col, A_combined):
    ratios = B_inv_b / pivot_col

    positive_ratios = ratios[ratios > 0]
    ratios = np.where(np.isinf(ratios), 0, ratios)
    print("\nDivision results: ", ratios)

    if len(positive_ratios) > 0:
        pivot_row = np.argmin(positive_ratios)
        return pivot_row
    else:
        print("The problem is unbounded.")
        return None

def revisedSimplex(B_inverse, W, B_inv_b, Cb_B_inv_b, pivot_col, pivot_row, pivot_col_index, pivot_head, B_indices, c_combined):
  pivot_element = pivot_col[pivot_row]

  B_inv_m, B_inv_n = B_inverse.shape

  for i in range(B_inv_m):
    for j in range(B_inv_n):
      if i == pivot_row:
        B_inverse[i][j] /= pivot_element
      else:
        B_inverse[i][j] = B_inverse[i][j] - ((pivot_col[i] * B_inverse[pivot_row][j])/ pivot_element)

  Cb_B_inv_b = Cb_B_inv_b - ((pivot_head * B_inv_b[pivot_row])/pivot_element)

  m,  = B_inv_b.shape

  for i in range(m):
    if i == pivot_row:
      B_inv_b[i] /= pivot_element
    else:
        B_inv_b[i] = B_inv_b[i] - ((pivot_col[i] * B_inv_b[pivot_row])/ pivot_element)
  
  new_base_variable = pivot_col_index

  B_indices = np.delete(B_indices, pivot_row)
  B_indices = np.append(B_indices, new_base_variable)

  n, = W.shape
  for i in range(n):
    W[i] = W[i] - ((pivot_head * B_inverse[pivot_row][i])/ pivot_element)

  return B_inverse, Cb_B_inv_b, B_inv_b, B_indices, W


# Example problem
'''
c = np.array([6, 14, 13])
A = np.array([[1, 2, 4], 
              [0.5, 2, 1]])
b = np.array([60, 24])
'''
c = np.array([3.8, 4.25])
A = np.array([[1, 0],
              [0, 1],
              [3, 5],
              [20, 10]])
b= np.array([28, 30, 180, 640])

tolerance = 10e-10

c_combined, A_combined = convert_inequality_to_equality(c, A, b)

print("\nInitial c:", c)
print("\nInitial A:")
print(A)
print("\nInitial b:", b)

print("\n----------------------------")
print("\nAfter converting the constraints to equality type constraints:")

print("\nNew c:", c_combined)
print("\nNew A:")
print(A_combined)

print("\n===========================FINDING BASE VARIABLES\n")
B_indices, B_inv, B_inv_b = initialize_simplex(c_combined, A_combined, b)

print("\n===========================CALCULATING W AND Cb * (B_inverse * b)\n")
W, Cb_B_inv_b = calculate_W_Cb_B_inv_b(c_combined, B_indices, B_inv, b)
print("W:", W)
print("\nCb * (B_inverse * b):", Cb_B_inv_b)

print("\n===========================FINDING PIVOT COLUMN\n")
pivot_col, pivot_head, pivot_col_index, pivot_head_values = find_pivot_column(W, c_combined, A_combined, B_indices, maximization=True)
print("\nPivot column values: ", pivot_col)
print("\nPivot Head:", pivot_head)
print("\nPivot column index: ", pivot_col_index)
print("\nPivot head values: ", pivot_head_values)

print("\n===========================FINDING PIVOT ROW\n")
pivot_row_index = find_pivot_row(B_inv_b, pivot_col, A_combined)
print("\nPivot row:", pivot_row_index)

maximization = True
while True:
  
  
  if maximization and np.all(np.greater_equal(pivot_head_values, -tolerance)):
    print("Optimal solution reached.")
    break
  if not maximization and np.all(np.less_equal(pivot_head_values, tolerance)):
    print("Optimal solution reached.")
    break

  print("\n------------------------------------\n")
  New_B_inverse, New_Cb_B_inv_b, New_B_inv_b, New_B_indices, New_W = revisedSimplex(B_inv, W, B_inv_b, Cb_B_inv_b, pivot_col, pivot_row_index, pivot_col_index, pivot_head, B_indices, c_combined)
  print("\nNew B_inv:")
  print(New_B_inverse)
  print("\nNew Cb * (B_inverse * b):")
  print(New_Cb_B_inv_b)
  print("\nNew B_inv * b:")
  print(New_B_inv_b)
  print("\nBasic variables indices: ", New_B_indices)
  print("\nNew W: ", New_W)

  pivot_col_result = find_pivot_column(New_W, c_combined, A_combined, New_B_indices, maximization=True)
  if pivot_col_result is None:
    break

  pivot_col, pivot_head, pivot_col_index, pivot_head_values = pivot_col_result
  print("\nPivot column values: ", pivot_col)
  print("\nPivot Head:", pivot_head)
  print("\nPivot column index: ", pivot_col_index)
  print("\nPivot head values: ", pivot_head_values)

  pivot_row_index = find_pivot_row(New_B_inv_b, pivot_col, A_combined)
  print("\nPivot row:", pivot_row_index)