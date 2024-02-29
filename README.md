# Senior-Thesis
This repository contains Python code for my thesis on the comparative analysis of classical algorithms and neural networks in solving the knapsack problem.

## Abstract
In my senior thesis, I evaluated four distinct algorithms—Dynamic Programming, Approximation Scheme, Nemhauser-Ullmann Algorithm, and Neural Network Algorithms—for the knapsack problem by implementing them in Python 3.11.3 and comparing their execution times and solution accuracies.

## Result
Within the range of instance sizes tested, the Nemhauser-Ullmann Algorithm consistently provided rapid and accurate solutions. Additionally, it was observed that the neural network algorithm, while taking longer to execute, was capable of discovering solutions with excellent accuracy for Approximation Scheme.


1 Comparison of Dynamic Programming and Nemhauser-Ullmann Algorithm

![Comparison of Execution Times](https://github.com/R-r-bic/Senior-Thesis/blob/main/Results%20of%20Execution%20Time.png?raw=true)

(a) When varying the number of items


| Number of items | Dynamic Programming | Nemhauser-Ullmann Algorithm |
|-----------------|---------------------|----------------------------|
| 10              | 0.0163              | 0.0185                     |
| 100             | 0.2011              | 0.2891                     |
| 300             | 1.6536              | 1.9724                     |
| 500             | 4.7753              | 3.3932                     |
| 1000            | 20.834              | 5.2586                     |

Table 1: Solution classes and execution times (seconds) for varying numbers of items

(b) Price range execution time comparison

| Price range   | Dynamic Programming | Nemhauser-Ullmann Algorithm |
|---------------|---------------------|-----------------------------|
| 1 ≤ ci ≤ 99   | 0.0163              | 0.0185                      |
| 100 ≤ ci ≤ 199| 0.5648              | 0.2382                      |
| 200 ≤ ci ≤ 299| 0.9284              | 0.2399                      |
| 300 ≤ ci ≤ 399| 1.3102              | 0.2364                      |

Table 2: Solution classes and execution times (seconds) for varying price ranges.


2 Comparison of Approximation Schemes and Neural Network Algorithms
(a) For 10 items

| Algorithm             | Execution Time (s) | Solution Accuracy |
|-----------------------|--------------------|-------------------|
| Approximation Schemes | 0.0008             | 0.64              |
| Neural Network        | 4.4161             | 0.78              |

Table 3: Execution time and solution accuracy for 10 items*

(b) For 100 items

| Algorithm             | Execution Time (s) | Solution Accuracy |
|-----------------------|--------------------|-------------------|
| Approximation Schemes | 0.3835             | 1.00              |
| Neural Network        | 71.8500            | 0.87              |

Table 4: Execution time and solution accuracy for 100 items*
