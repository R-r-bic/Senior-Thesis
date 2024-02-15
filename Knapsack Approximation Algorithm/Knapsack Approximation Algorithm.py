import numpy as np
import time

start_time = time.time()

# Solution to the knapsack problem using dynamic programming
def dynamic_programming_knapsack(weights, values, W, C):
    n = len(weights)
    C = int(C)
    x = [[0 if k == 0 else float('inf') for k in range(C + 1)] for _ in range(n + 1)]
    s = [[0 for _ in range(C + 1)] for _ in range(n + 1)]

    for j in range(1, n + 1):
        for k in range(C + 1):
            x[j][k] = x[j - 1][k]
            s[j][k] = 0
            if k >= values[j - 1] and x[j - 1][k - values[j - 1]] + weights[j - 1] <= min(W, x[j][k]):
                x[j][k] = x[j - 1][k - values[j - 1]] + weights[j - 1]
                s[j][k] = 1

    S2 = set()
    k = max(i for i in range(C + 1) if x[n][i] < float('inf'))
    for j in range(n, 0, -1):
        if s[j][k] == 1:
            S2.add(j - 1)
            k -= values[j - 1]
    
    return S2

# Knapsack approximation scheme
def knapsack_approximation_scheme(values, weights, W, epsilon):
    # Step 1: Find an initial solution S1 based on the greedy method
    S1 = set()
    c_S1 = 0
    for i, (value, weight) in enumerate(sorted(zip(values, weights), key=lambda x: -x[0]/x[1])):
        if sum(weights[j] for j in S1) + weight <= W:
            S1.add(i)
            c_S1 += value
    if c_S1 == 0:
        return S1  # If c(S1) is 0, then it is the optimal solution
    
    # Step 2: Calculate t and the modified value c_j
    n = len(values)
    t = max(1, epsilon * c_S1 / n)
    modified_values = [int(value / t) for value in values]
    
    # Step 3: Apply dynamic programming knapsack algorithm using modified values and C
    C = 2 * sum(modified_values) // t  # Calculation of C
    S2 = dynamic_programming_knapsack(weights, modified_values, W, C)
    
    # Step 4: Choose the better solution between S1 and S2 based on the original values
    c_S2 = sum(values[i] for i in S2)
    return S1 if c_S1 >= c_S2 else S2

# Load dataset from CSV file
from Dataset_generate_dynamic10 import load_dataset_from_csv

dataset = load_dataset_from_csv('knapsack_dataset.csv')
num_sets = len(dataset)

# Example usage with placeholder dataset and function
epsilon = 0.1  # This can be adjusted as needed
for weights, values, capacity in dataset:
    W = int(capacity)
    solution = knapsack_approximation_scheme(values, weights, W, epsilon)
    sorted_solution = sorted(solution) 
    print(sorted_solution)

# Record end time of the execution
end_time = time.time()

# Calculate total execution time
total_execution_time = end_time - start_time

# Calculate average execution time
average_execution_time = total_execution_time / num_sets

# Display average execution time
print("Average execution time:", average_execution_time, "seconds")

from dynamic import knapsack_dynamic_programming_fixed

# Evaluation of the solution against the optimal solution
def evaluate_solution(solution, weights, values, capacity, optimal_solution):
    true_positives = sum(1 for i in solution if i in optimal_solution)
    # Number of true negatives (correctly not included items)
    true_negatives = sum(1 for i in range(len(weights)) if i not in solution and i not in optimal_solution)
    # Total number of predictions (all items)
    total_predictions = len(weights)
    # Calculation of binary accuracy
    binary_accuracy = (true_positives + true_negatives) / total_predictions

    # Space violation
    total_weight = sum(weights[i] for i in solution)
    space_violation = max(total_weight - capacity, 0)

    # Overpricing
    overpricing = sum(values[i] for i in solution)

    # Pick count
    pick_count = len(solution)

    return binary_accuracy, space_violation, overpricing, pick_count

# Initialize variables for averages
total_binary_accuracy = 0
total_space_violation = 0
total_overpricing = 0
total_pick_count = 0
total_value_ratio = 0
num_instances = 0

# Loop through each instance in the dataset
for i, (weights, values, capacity) in enumerate(dataset):
    solution = knapsack_approximation_scheme(values, weights, int(capacity), epsilon=0.5)
    optimal_solution = knapsack_dynamic_programming_fixed(weights, values, capacity)
    optimal_solution_indices = knapsack_dynamic_programming_fixed(weights, values, capacity)

    total_value = sum(values[i] for i in optimal_solution_indices)

    print("Total value of the optimal solution:", total_value)
    
    if not optimal_solution:
        print(f"Instance {i + 1}: Optimal solution is empty, cannot evaluate.")
        continue

    binary_accuracy, space_violation, overpricing, pick_count = evaluate_solution(solution, weights, values, capacity, optimal_solution)
    print(f"Instance {i + 1}: Binary Accuracy: {binary_accuracy}, Space Violation: {space_violation}, Overpricing: {overpricing}, Pick Count: {pick_count}")

    total_binary_accuracy += binary_accuracy
    total_space_violation += space_violation
    total_overpricing += overpricing
    total_pick_count += pick_count

    # New evaluation metric: Value ratio
    solution_value = sum(values[i] for i in solution)
    optimal_value = sum(values[i] for i in optimal_solution)
    value_ratio = solution_value / optimal_value if optimal_value > 0 else 0
    total_value_ratio += value_ratio

    num_instances += 1

# Calculate averages
average_binary_accuracy = total_binary_accuracy / num_instances
average_space_violation = total_space_violation / num_instances
average_overpricing = total_overpricing / num_instances
average_pick_count = total_pick_count / num_instances
average_value_ratio = total_value_ratio / num_instances

print(f"Average Binary Accuracy: {average_binary_accuracy}")
print(f"Average Space Violation: {average_space_violation}")
print(f"Average Overpricing: {average_overpricing}")
print(f"Average Pick Count: {average_pick_count}")
print(f"Average Value Ratio: {average_value_ratio}")

total_sum_value = 0  # Variable for storing the total sum of values

for i, (weights, values, capacity) in enumerate(dataset):
    sum_value = sum(values)
    print(f"Instance {i + 1}: Total Value = {sum_value}")

    total_sum_value += sum_value

average_sum_value = total_sum_value / len(dataset)
print("Average Total Value of Instances:", average_sum_value)
