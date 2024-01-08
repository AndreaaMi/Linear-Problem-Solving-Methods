import numpy as np


def print_matrix(costs,supply,demand):
    m, n = costs.shape

    for i in range(m):
        for j in range(n):
            print("{:4}".format(costs[i][j]), end=' ')
        print("{:4}".format(supply[i]))

    for j in range(n):
        print("{:4}".format(demand[j]), end=' ')


def print_solution_matrix(solution, supply, demand):
    m, n = solution.shape

    for i in range(m):
        for j in range(n):
            print("{:4}".format(solution[i][j]), end=' ')
        print("{:4}".format(supply[i]))

    for j in range(n):
        print("{:4}".format(demand[j]), end=' ')


def least_cost_method(costs, supplies, demands):
    m, n = costs.shape
    solution = np.zeros((m, n), dtype=int)

    while np.any(supplies > 0) and np.any(demands > 0):
        min_cost = np.inf
        min_i, min_j = -1, -1

        for i in range(m):
            for j in range(n):
                if supplies[i] > 0 and demands[j] > 0 and costs[i, j] < min_cost:
                    min_cost = costs[i, j]
                    min_i, min_j = i, j

        if min_i == -1 or min_j == -1:
            break

        quantity = min(supplies[min_i], demands[min_j])
        solution[min_i, min_j] = quantity

        supplies[min_i] -= quantity
        demands[min_j] -= quantity

    solution = np.where(solution == 0, -1, solution)
    return solution


def check_if_balanced(costs, demand, supply):
    m, n = costs.shape

    # Check if the problem is unbalanced
    if np.sum(supply) != np.sum(demand):
        print("The transportation problem is unbalanced!\nTotal supply:", np.sum(supply),"\nTotal demand:", np.sum(demand))

        adjustment_value = np.abs(np.sum(supply) - np.sum(demand))
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
    check_if_balanced(costs, demand, supply)

    print("\nInitial Costs Matrix:")
    print_matrix(costs,supply,demand)
    print("\n")
    initial_solution =  np.zeros_like(costs)
    initial_solution = least_cost_method(costs, supply, demand)

    iteration = 1
    while True:
        print(f"\n-------- {iteration}. Iteration --------\n")
        print("\nCurrent Solution Matrix:")

        print_solution_matrix(initial_solution, supply, demand)

        U, V = uv(costs, initial_solution)
        print("\n\nU:", U, "  V:", V, "\n" )

        new_costs = calculate_new_costs(costs, U, V, initial_solution)
        print("New Costs:")
        print_matrix(new_costs, supply, demand)

        if np.all(new_costs >= 0):
            print("\n\n----------------------------------")
            break

        i, j = find_most_negative(new_costs)
        start_contour_cell = (i, j)
        contour = form_contour(initial_solution, start_contour_cell, new_costs)

        print("\n\nContour:")
        print(contour)

        temp_solution = np.where(initial_solution == -1, 0, initial_solution)
        min_value = min(temp_solution[cell] for cell in contour[1:-1])
        initial_solution = update_solution(initial_solution, contour, min_value)

        print("\nUpdated Solution:")
        print_solution_matrix(initial_solution, supply, demand)
        print("\n")

        iteration += 1

    print("Final Solution Matrix:")
    print_solution_matrix(initial_solution, supply, demand)

    total_cost = np.sum(initial_solution * costs)
    print("\n\nMinimal Cost: Z =", total_cost, "!\n")


def uv(costs, solution):
    m, n = costs.shape

    U = np.full(m, '*', dtype='object')
    V = np.full(n, '*', dtype='object')

    row_with_max_elements = np.argmax(np.sum(solution != -1, axis=1))
    U[row_with_max_elements] = 0

    while '*' in U or '*' in V:
        for i in range(m):
            for j in range(n):
                if solution[i, j] != -1:
                    if U[i] == '*' and V[j] != '*':
                        U[i] = int(costs[i, j]) - int(V[j])
                    elif U[i] != '*' and V[j] == '*':
                        V[j] = int(costs[i, j]) - int(U[i])

        for j in range(n):
            non_zero_elements = solution[:, j].nonzero()[0]
            if len(non_zero_elements) == 1:
                i = non_zero_elements[0]
                if U[i] == '*' and V[j] != '*':
                    U[i] = int(costs[i, j]) - int(V[j])
                elif U[i] != '*' and V[j] == '*':
                    V[j] = int(costs[i, j]) - int(U[i])

    return U, V


def form_contour(solution_matrix, start_contour_cell, new_costs):
    m, n = solution_matrix.shape
    visited_cells = set()
    contour = []

    def find_cycle(cell):
        visited_cells.add(cell)
        contour.append(cell)

        i, j = cell
        for col in range(n):
            next_cell = (i, col)
            if solution_matrix[next_cell] != -1 and next_cell not in visited_cells:
                find_cycle(next_cell)

        for row in range(m):
            next_cell = (row, j)
            if solution_matrix[next_cell] != -1 and next_cell not in visited_cells:
                find_cycle(next_cell)

    find_cycle(start_contour_cell)
    contour.append(contour[0]) 

    pruned_path = [contour[0]]
    for i in range(1, len(contour)-1):
        if contour[i][0] == contour[i+1][0] and contour[i][1] != contour[i+1][1]:
            col = contour[i+1][1] if new_costs[contour[i+1][0], contour[i+1][1]] == 0 else contour[i][1]
            pruned_path.append((contour[i][0], col))
        elif contour[i][1] == contour[i+1][1] and contour[i][0] != contour[i+1][0]:
            row = contour[i+1][0] if new_costs[contour[i+1][0], contour[i+1][1]] == 0 else contour[i][0]
            pruned_path.append((row, contour[i][1]))

    for i in range(3, len(pruned_path)):
        current_cell = pruned_path[i]
        base_cell = pruned_path[0]

        if current_cell[0] == base_cell[0] or current_cell[1] == base_cell[1]:
            final_contour = pruned_path[:i+1].copy()
            break

    return final_contour


def calculate_new_costs(costs, U, V, solution):
    m, n = costs.shape
    new_costs = np.zeros_like(solution, dtype=int)
    tmp_solution = np.where(solution == 0, -1, solution)

    for i in range(m):
        for j in range(n):
            if tmp_solution[i, j] == -1:
                new_costs[i, j] = costs[i, j] - U[i] - V[j]

    return new_costs


def find_most_negative(new_costs):
    min_penalty = np.min(new_costs)
    indices = np.where(new_costs == min_penalty)
    return indices[0][0], indices[1][0]


def update_solution(solution, contour, min_value):
    solution = np.where(solution == -1, 0, solution)
    visited_nodes = set()

    for i in range(len(contour)):
        x, y = contour[i]

        if i % 2 == 0 and (x, y) not in visited_nodes:
            solution[x, y] += min_value
            visited_nodes.add((x, y))
        elif i % 2 != 0 and (x, y) not in visited_nodes:
            solution[x, y] -= min_value
            visited_nodes.add((x, y))

    return solution


# Example usage
costs = np.array([[4,9,2],
                  [7,5,3],
                  [1,6,3],
                  [3,2,8]])
supply = np.array([150,350,300,200])
demand = np.array([250,350,400])

iterative_transportation_method(costs, demand, supply)
