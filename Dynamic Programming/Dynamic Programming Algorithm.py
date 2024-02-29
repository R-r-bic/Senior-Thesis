import time
start_time = time.time()
 
def knapsack_dynamic_programming_fixed(weights, values, capacity):
    # Convert weights and values to integer type
    weights = [int(w) for w in weights]
    values = [int(v) for v in values]

    n = len(weights)
    C = int(sum(values))

    # Initialize the x and s matrices
    x = [[0 if k == 0 else float('inf') for k in range(C + 1)] for _ in range(n + 1)]
    s = [[0 for _ in range(C + 1)] for _ in range(n + 1)]

    # Dynamic programming to fill x and s matrices
    for j in range(1, n + 1):
        for k in range(0, C + 1):
            x[j][k] = x[j - 1][k]
            s[j][k] = 0
        for k in range(values[j - 1], C + 1):
            if x[j - 1][k - values[j - 1]] + weights[j - 1] <= min(capacity, x[j][k]):
                x[j][k] = x[j - 1][k - values[j - 1]] + weights[j - 1]
                s[j][k] = 1

    # Reconstruct the solution
    k = max(i for i in range(C + 1) if x[n][i] < float('inf'))
    selected_items = set()
    for j in range(n, 0, -1):
        if s[j][k] == 1:
            selected_items.add(j - 1)
            k -= values[j - 1]

    return selected_items


from Dataset_generate_dynamic10 import load_dataset_from_csv

# Load the dataset from a CSV file
dataset = load_dataset_from_csv('knapsack_dataset.csv')

# Number of sets
num_sets = len(dataset)


for i, (weights, values, capacity) in enumerate(dataset):

    solution = knapsack_dynamic_programming_fixed(weights, values, capacity)
    sorted_solution = sorted(solution) 
    
    print("Selected items:", sorted_solution)
    
# Record the total execution end time
end_time = time.time()
# Calculate the total execution time
total_execution_time = end_time - start_time

# Calculate the average execution time
average_execution_time = total_execution_time / num_sets

# Display the average execution time
print("Average execution time:", average_execution_time, "seconds")

