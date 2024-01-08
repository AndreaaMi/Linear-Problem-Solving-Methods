import numpy as np


def print_matrix(costs,supply,demand):
    m, n = costs.shape

    for i in range(m):
        for j in range(n):
            print("{:4}".format(costs[i][j]), end=' ')
        print("{:4}".format(supply[i]))

    for j in range(n):
        print("{:4}".format(demand[j]), end=' ')


def north_west_corner(supply, demand, initial_solution):
    m, n = costs.shape
    i = j = 0
    while i < m and j < n:
        allocation = min(supply[i], demand[j])
        initial_solution[i][j] += allocation
        supply[i] -= allocation
        demand[j] -= allocation

        # Cross out the row or column if supply or demand becomes zero
        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1

def print_solution_matrix(solution, supply, demand):
    m, n = solution.shape

    for i in range(m):
        for j in range(n):
            print("{:4}".format(solution[i][j]), end=' ')
        print("{:4}".format(supply[i]))

    for j in range(n):
        print("{:4}".format(demand[j]), end=' ')

def check_if_balanced(costs, demand, supply):
    m, n = costs.shape

    # Check if the problem is unbalanced
    if np.sum(supply) != np.sum(demand):
        print("The transportation problem is unbalanced!\nTotal supply:", np.sum(supply),"\nTotal demand:", np.sum(demand))

        # Calculate the necessary adjustment value
        adjustment_value = np.abs(np.sum(supply) - np.sum(demand))

        # Add dummy row or column based on whether supply or demand is greater
        if np.sum(supply) > np.sum(demand):
            demand = np.append(demand, adjustment_value)
            costs = np.column_stack((costs, np.zeros(m)))
        elif np.sum(supply) < np.sum(demand):
            supply = np.append(supply, adjustment_value)
            costs = np.row_stack((costs, np.zeros(n)))

        print("Adjusting to balance the problem...")
        print("Adjusted supply:", supply)
        print("Adjusted demand:", demand)
    else:
        print("The transportation problem is balanced!")
        
def iterative_transportation_method(costs, demand, supply):
    m, n = costs.shape

    check_if_balanced(costs, demand, supply)

    print("\nInitial Matrix:")
    print_matrix(costs,supply,demand)

    initial_solution = np.zeros((m, n), dtype=int)

    iteration = 1
    while True:
        print(f"\n-------- {iteration}. Iteration --------\n")
        north_west_corner(supply, demand, initial_solution)

        # Print the current solution matrix
        print("\nCurrent Solution Matrix:")
        print_solution_matrix(initial_solution, supply, demand)

        sum_cost = np.sum(initial_solution * costs)

        print("\n\nCurrent Minimum cost is", sum_cost, "!\n")

        # Implement UV method, calculate penalties, and find closed path
        U, V = uv_method(costs, initial_solution)
        penalties = calculate_penalties(costs, U, V, initial_solution)
        i, j = find_most_negative(penalties)
        new_basic_cell = (i, j)
        closed_path = form_closed_path(initial_solution, new_basic_cell)

        print("\nClosed Path:")
        print(closed_path)

        min_value = min(initial_solution[cell] for cell in closed_path[1:-1])
        initial_solution = update_solution(initial_solution, closed_path, min_value)

        print("\nUpdated Initial Solution:")
        print_solution_matrix(initial_solution, supply, demand)


        if np.all(penalties >= 0):
            break

        iteration += 1
        print("\n------------------------\n")

    print("\n\nFinal Solution Matrix:")
    print_solution_matrix(initial_solution, supply, demand)


    print("\n\nFinal Minimum cost is", sum_cost, "!")


def uv_method(costs, solution):
    m, n = costs.shape
    U = np.zeros(m)
    V = np.zeros(n)

    while True:
        for i in range(m):
            for j in range(n):
                if solution[i, j] != 0:
                    if U[i] == 0 and V[j] == 0:
                        U[i] = costs[i, j]
                    else:
                        V[j] = costs[i, j] - U[i]

        for j in range(n):
            non_zero_elements = solution[:, j].nonzero()[0]
            if len(non_zero_elements) == 1:
                i = non_zero_elements[0]
                U[i] = costs[i, j] - V[j]
        break

    return U, V


def calculate_penalties(costs, U, V, solution):
    m, n = costs.shape
    penalties = np.zeros_like(solution, dtype=int)

    for i in range(m):
        for j in range(n):
            if solution[i, j] == 0:
                penalties[i, j] = U[i] + V[j] - costs[i, j]

    return penalties


def find_most_negative(penalties):
    min_penalty = np.min(penalties)
    indices = np.where(penalties == min_penalty)
    return indices[0][0], indices[1][0]


def update_solution(solution, closed_path, min_value):
    visited_nodes = set()

    for i in range(len(closed_path)):
        x, y = closed_path[i]

        # Add condition to check whether the node is already visited
        if i % 2 == 0 and (x, y) not in visited_nodes:
            solution[x, y] += min_value
            visited_nodes.add((x, y))
        elif i % 2 != 0 and (x, y) not in visited_nodes:
            solution[x, y] -= min_value
            visited_nodes.add((x, y))

    return solution


def form_closed_path(solution_matrix, new_basic_cell):
    m, n = solution_matrix.shape
    visited_cells = set()
    closed_path = []

    def find_cycle(cell):
        visited_cells.add(cell)
        closed_path.append(cell)

        i, j = cell
        for col in range(n):
            next_cell = (i, col)
            if solution_matrix[next_cell] != 0 and next_cell not in visited_cells:
                find_cycle(next_cell)

        for row in range(m):
            next_cell = (row, j)
            if solution_matrix[next_cell] != 0 and next_cell not in visited_cells:
                find_cycle(next_cell)

    find_cycle(new_basic_cell)
    closed_path.append(closed_path[0])

    prev_i, prev_j = closed_path[0]
    k = 1
    while k < len(closed_path) - 1:
        next_i, next_j = closed_path[k]
        if next_i != prev_i and next_j != prev_j:
            closed_path.pop(k)
            prev_i, prev_j = next_i, next_j
        else:
            k += 1
            prev_i, prev_j = next_i, next_j

    return closed_path


# Example usage
costs = np.array([[10, 12, 0],
                  [8, 4, 3],
                  [6, 9, 4],
                  [7, 8, 5]])
supply = np.array([20, 30, 20, 10])
demand = np.array([10, 40, 30])

iterative_transportation_method(costs, demand, supply)
