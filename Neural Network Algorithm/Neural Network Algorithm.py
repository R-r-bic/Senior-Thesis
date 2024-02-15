import os
import numpy as np
import tensorflow as tf
import time

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



def create_knapsack(item_count=10, seed=None):
    np.random.seed(seed)  # Set the seed value
    x_weights = np.random.randint(1, 45, item_count)
    x_prices = np.random.randint(1, 99, item_count)
    x_capacity = np.random.randint(1, 99)
    y = list(knapsack_dynamic_programming_fixed(x_weights, x_prices, x_capacity))
    return x_weights, x_prices, x_capacity,y


def knapsack_loss(input_weights, input_prices, cvc):
    def loss(y_true, y_pred):
        picks = y_pred
        violation = tf.keras.backend.maximum(tf.keras.backend.batch_dot(picks, input_weights, 1) - 1, 0)
        price = tf.keras.backend.batch_dot(picks, input_prices, 1)
        return cvc * violation - price

    return loss


def metric_overprice(input_prices):
    def overpricing(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.batch_dot(y_pred, input_prices, 1) - tf.keras.backend.batch_dot(y_true, input_prices, 1))

    return overpricing


def metric_space_violation(input_weights):
    def space_violation(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.maximum(tf.keras.backend.batch_dot(y_pred, input_weights, 1) - 1, 0))

    return space_violation


def metric_pick_count():
    def pick_count(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.sum(y_pred, -1))

    return pick_count


def unsupervised_model(cvc=5.75, item_count=10):
    input_weights = tf.keras.Input((item_count,))
    input_prices = tf.keras.Input((item_count,))
    inputs_concat = tf.keras.layers.Concatenate()([input_weights, input_prices])
    picks = tf.keras.layers.Dense(item_count ** 2 + item_count * 2, activation="sigmoid")(inputs_concat)
    picks = tf.keras.layers.Dense(item_count, activation="sigmoid")(picks)
    model = tf.keras.Model(inputs=[input_weights, input_prices], outputs=[picks])
    model.compile("adam",
                  knapsack_loss(input_weights, input_prices, cvc),
                  metrics=[tf.keras.metrics.binary_accuracy, metric_space_violation(input_weights),
                           metric_overprice(input_prices), metric_pick_count()])
    return model



def create_knapsack_dataset(count, item_count=10, seed=None):
    x1 = []
    x2 = []
    y = []
    original_weights = []  # Original weights
    original_prices = []   # Original values
    capacities = []        # Capacities
    for i in range(count):
        x_weights, x_prices, x_capacity, answer = create_knapsack(item_count, seed=seed+i)
        x1.append(x_weights / x_capacity)
        x2.append(x_prices / x_prices.max())

        # Convert answer to a fixed-length binary vector
        answer_vector = [1 if j in answer else 0 for j in range(item_count)]
        y.append(answer_vector)
        
        original_weights.append(x_weights)
        original_prices.append(x_prices)
        capacities.append(x_capacity)

    # Convert lists to NumPy arrays and return
    return np.array(x1), np.array(x2), np.array(y), original_weights, original_prices, capacities



def train_knapsack(model, train_x, train_y, test_x, test_y):
    if os.path.exists("best_model.h5"): os.remove("best_model.h5")
    model.fit(train_x, train_y, epochs=100, verbose=1,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor="binary_accuracy", save_best_only=True,
                                                     save_weights_only=True)])
    model.load_weights("best_model.h5")
    train_results = model.evaluate(train_x, train_y, 64, 0)
    test_results = model.evaluate(test_x, test_y, 64, 0)
    print("Model results(Train/Test):")
    print(f"Loss:               {train_results[0]:.2f} / {test_results[0]:.2f}")
    print(f"Binary accuracy:    {train_results[1]:.2f} / {test_results[1]:.2f}")
    print(f"Space violation:    {train_results[2]:.2f} / {test_results[2]:.2f}")
    print(f"Overpricing:        {train_results[3]:.2f} / {test_results[3]:.2f}")
    print(f"Pick count:         {train_results[4]:.2f} / {test_results[4]:.2f}")


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    x1_train, x2_train, train_y, test_weights, test_prices, test_capacities  = create_knapsack_dataset(1000, item_count=10, seed=1000)
    x1_test, x2_test, test_y, test_weights, test_prices, test_capacities = create_knapsack_dataset(20, item_count=10, seed=0)

    # Combine training and testing datasets
    train_x = [x1_train, x2_train]
    test_x = [x1_test, x2_test]

    start_time = time.time()
    # Define and train the model
    model = unsupervised_model(item_count=10)
    train_knapsack(model, train_x, train_y, test_x, test_y)
    
    end_time = time.time()

    # Calculate execution time
    training_duration = end_time - start_time
    print(f"Training execution time: {training_duration} seconds")
    
for i in range(len(test_weights)):
    print(f"Test instance {i + 1}:")
    print(f"Original weights: {test_weights[i]}")
    print(f"Original values: {test_prices[i]}")
    print(f"Capacity: {test_capacities[i]}")
    print(f"Selected items: {test_y[i]}")
    print()
    
def calculate_weighted_solution_value(model, test_x, test_prices):
    predicted_probabilities = model.predict(test_x)
    weighted_values = []

    for i in range(len(predicted_probabilities)):
        weighted_value = sum(predicted_probabilities[i][j] * test_prices[i][j] for j in range(len(test_prices[i])))
        weighted_values.append(weighted_value)

    return weighted_values

def calculate_optimal_solution_value(optimal_solutions, test_prices):
    optimal_values = [sum(optimal_solutions[i][j] * test_prices[i][j] for j in range(len(test_prices[i]))) for i in range(len(optimal_solutions))]
    return optimal_values

weighted_solution_values = calculate_weighted_solution_value(model, test_x, test_prices)
optimal_solution_values = calculate_optimal_solution_value(test_y, test_prices)

value_ratios = [weighted / optimal if optimal > 0 else 0 for weighted, optimal in zip(weighted_solution_values, optimal_solution_values)]
average_value_ratio = sum(value_ratios) / len(value_ratios) if value_ratios else 0

print(f"Average Value Ratio: {average_value_ratio}")


