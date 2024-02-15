import numpy as np
import csv

def create_knapsack(item_count=10, seed=None):
    np.random.seed(seed)  # シード値の設定
    x_weights = np.random.randint(1, 45, item_count)
    x_prices = np.random.randint(1, 99, item_count)
    x_capacity = np.random.randint(1, 99)
    return x_weights, x_prices, x_capacity


def create_knapsack_dataset(count, item_count=10, seed=None, normalize=False):
    x1 = []
    x2 = []
    capacities = []  # ナップサックの容量を保存するリスト
    for i in range(count):
        x_weights, x_prices, x_capacity = create_knapsack(item_count, seed=seed+i)
        if normalize:
            x1.append(x_weights / x_capacity)
            x2.append(x_prices / x_prices.max())
        else:
            x1.append(x_weights)
            x2.append(x_prices)
        capacities.append(x_capacity)   # 容量をリストに追加

        
    return [np.array(x1), np.array(x2)], np.array(capacities)
            
def save_dataset_to_csv(x, capacities, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Weights', 'Prices', 'Capacity'])
        for weights, prices, capacity in zip(x[0], x[1], capacities):
            weights_str = ' '.join(map(str, weights))
            prices_str = ' '.join(map(str, prices))
            writer.writerow([weights_str, prices_str, capacity])

# データセットを生成
x, capacities = create_knapsack_dataset(20, 10, seed=0, normalize=False)

# データセットをCSVに保存
save_dataset_to_csv(x, capacities, 'knapsack_dataset.csv')


def load_dataset_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダー行をスキップ
        dataset = []
        for row in reader:
            weights = list(map(float, row[0].split()))
            prices = list(map(float, row[1].split()))
            capacity = float(row[2])
            dataset.append((weights, prices, capacity))
        return dataset

# CSVからデータセットを読み込む
dataset = load_dataset_from_csv('knapsack_dataset.csv')



# 生成されたデータセットを表示　　プラスで大きさは大丈夫か　　
for i in range(20):
    print(f"インスタンス {i + 1}:")
    print(f"重量: {x[0][i]}")
    print(f"価値: {x[1][i]}")
    print(f"容量: {capacities[i]}")
    print()


