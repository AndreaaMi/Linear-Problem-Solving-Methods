def transportation_problem_north_west_corner(costs, demand, supply):
    m = len(supply)
    n = len(demand)

    # Check if the problem is unbalanced
    if sum(supply) != sum(demand):
        print("The transportation problem is unbalanced.")
        print("Total supply:", sum(supply))
        print("Total demand:", sum(demand))

        # Calculate the necessary adjustment value
        adjustment_value = abs(sum(supply) - sum(demand))

        # Add dummy row or column based on whether supply or demand is greater
        if sum(supply) > sum(demand):
            demand.append(adjustment_value)
            costs.append([0] * n)
        elif sum(supply) < sum(demand):
            supply.append(adjustment_value)
            for row in costs:
                row.append(0)

        print("Adjusting to balance the problem...")
        print("Adjusted supply:", supply)
        print("Adjusted demand:", demand)

    print("\nInitial Matrix:")
    for i in range(m):
        for j in range(n):
            print("{:4}".format(costs[i][j]), end=' ')
        print("{:4}".format(supply[i]))

    for j in range(n):
        print("{:4}".format(demand[j]), end=' ')

    # Initialize the initial solution matrix with zeros
    initial_solution = [[0] * n for _ in range(m)]

    # Find the initial solution using North West Corner Rule
    i = j = 0
    while i < m and j < n:
        allocation = min(supply[i], demand[j])
        initial_solution[i][j] = allocation
        supply[i] -= allocation
        demand[j] -= allocation

        # Cross out the row or column if supply or demand becomes zero
        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1

    # Print the initial solution matrix
    print("\n\nInitial Solution Matrix:")
    for i in range(m):
        for j in range(n):
            print("{:4}".format(initial_solution[i][j]), end=' ')
        print("{:4}".format(supply[i]))

    for j in range(n):
        print("{:4}".format(demand[j]), end=' ')

    sum_cost = 0
    for i in range(m):
        for j in range(n):
            sum_cost += initial_solution[i][j] * costs[i][j]

    print("\n\nInitial Minimum cost is", sum_cost, "!")


# Example usage
costs = [[11, 13, 17, 14],
         [16, 18, 14, 10],
         [21, 24, 13, 10]]
supply = [250, 300, 430]
demand = [200, 225, 275, 300]

transportation_problem_north_west_corner(costs, demand, supply)
