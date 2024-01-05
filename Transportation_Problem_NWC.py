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

    print("\nCost Matrix:\n")
    for i in range(m):
        for j in range(n):
            print(" ", costs[i][j], end=' ')
        print(supply[i])

    for j in range(n):
        print("", demand[j], end=' ')

    sum_cost = 0
    i = j = 0
    while i < m and j < n:
        if supply[i] < demand[j]:
            sum_cost += costs[i][j] * supply[i]
            demand[j] -= supply[i]
            i += 1
        elif supply[i] > demand[j]:
            sum_cost += costs[i][j] * demand[j]
            supply[i] -= demand[j]
            j += 1
        elif supply[i] == demand[j]:
            sum_cost += costs[i][j] * demand[j]
            i += 1
            j += 1

    print("\n\nMinimum cost is", sum_cost, "!")

# Example usage
costs = [[11, 13, 17, 14],
         [16, 18, 14, 10],
         [21, 24, 13, 10]]
supply = [250, 300, 430]
demand = [200, 225, 275, 300]

transportation_problem_north_west_corner(costs, demand, supply)
