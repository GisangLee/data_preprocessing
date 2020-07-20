import numpy as np
import pandas as pd

# 데이터
x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
y_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

# hypothesis
W = np.random.rand(1, 1)
b = np.random.rand(1)
print("W = {}, b={}\n".format(W, b))


# cost loss
def loss_func(x, t):
    y = np.dot(x, W) + b
    return (np.sum((t - y) ** 2)) / (len(x))


# numerical_derivative
def multi_var_numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()
    return grad


# 손실함수 값 계산 (cost function)
def error_val(x, t):
    y = np.dot(x, W) + b
    return (np.sum((t - y) ** 2)) / (len(x))


# 학습 후 임의 데이터에 대해 예측
def predict(x):
    y = np.dot(x, W) + b
    print(y)
    return y


# learning rate
learning_rate = 1e-2

f = lambda x: loss_func(x_data, y_data)


# W, b
print("initial error value = {0}, W = {1}, b = {2} \n".format(error_val(x_data, y_data), W, b))

for step in range(8001):
    W -= learning_rate * multi_var_numerical_derivative(f, W)
    b -= learning_rate * multi_var_numerical_derivative(f, b)
    if step % 200 == 0:
        print("step = {0}, error_val = {1}, W = {2}, b = {3}\n".format(step, error_val(x_data, y_data), W, b))

predict(45)

