import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda
from keras.metrics import binary_accuracy
from keras.optimizers import Adam

# ブルートフォース法を用いて最適解を求める関数
def brute_force_knapsack(x_weights, x_prices, x_capacity):
    item_count = len(x_weights)
    best_price = -1
    best_picks = None
    for p in range(2 ** item_count):
        #2 ** item_countはアイテムの組み合わせの総数を計算。すべてのアイテムに関しては選ばれるか、選ばれないかのどちらかがあるからitem_countの２乗個通りがある→選ぶ系の考え方の問題はこれを使うとやりやすい
        picks = np.array([int(c) for c in f"{p:0{item_count}b}"])
        #アイテムの数の配列を並べて、それを２進法（必要か、必要でないかを判断する）で表記して、アイテムを取るかどうかを表現している（つまりアイテムを使うかどうかをわかるためのコードを作り出した）
        #０は０埋めを表現しており、開いた桁をゼロで埋めることを指示する
        #b：これはbinary formatを表していて、数値を２進数の文字列としてフォーマットすることを指示する
        total_price = np.dot(x_prices, picks)
        # picksは１か０の配列になっているのだから、内積をかけても価値の総和になるだけ
        total_weight = np.dot(x_weights, picks)
        #これも同様で、重さの総和になるだけ
        if total_weight <= x_capacity and total_price > best_price:
            best_price = total_price
            best_picks = picks
    return best_price, best_picks    


# ナップサック問題のインスタンスを生成する関数
def create_knapsack(item_count=5):
    x_weights = np.random.randint(1, 15, item_count)
    x_prices = np.random.randint(1, 10, item_count)
    x_capacity = np.random.randint(15, 50)
    _, y_best_picks = brute_force_knapsack(x_weights, x_prices, x_capacity)
    return x_weights, x_prices, x_capacity, y_best_picks

# データセットを生成する関数
def create_knapsack_dataset(count, item_count=5):
    x_weights, x_prices, x_capacities, y_picks = [], [], [], []
    for _ in range(count):
        weights, prices, capacity, picks = create_knapsack(item_count)
        x_weights.append(weights)
        x_prices.append(prices)
        x_capacities.append([capacity])
        y_picks.append(picks)
    return [np.array(x_weights), np.array(x_prices), np.array(x_capacities)], np.array(y_picks)

# ニューラルネットワークモデルを定義する関数
def knapsack_neural_network_model(input_dim):
    input_weights = Input(shape=(input_dim,))
    input_prices = Input(shape=(input_dim,))
    input_capacity = Input(shape=(1,))

    # 結合層
    concatenated = Concatenate()([input_weights, input_prices, input_capacity])

    # 隠れ層
    hidden = Dense(128, activation="relu")(concatenated)
    hidden = Dense(64, activation="relu")(hidden)

    # 出力層
    output = Dense(input_dim, activation="sigmoid")(hidden)

    # モデルの定義
    model = Model(inputs=[input_weights, input_prices, input_capacity], outputs=output)

    # コンパイル
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# データセットの生成
train_x, train_y = create_knapsack_dataset(15000, item_count=5)
test_x, test_y = create_knapsack_dataset(300, item_count=5)

# モデルの定義と訓練
model = knapsack_neural_network_model(5)
model.fit(train_x, train_y, epochs=10, batch_size=32)

# テストデータでの評価
test_results = model.evaluate(test_x, test_y)
print("Test Results:", test_results)



