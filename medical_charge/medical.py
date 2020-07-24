import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

db = pd.read_csv("../data_sets/medical_charge.csv")
db.smoker = db.smoker.map({"yes": 1, "no": 0})
db.sex = db.sex.map({"female": 1, "male": 0})
db.region = db.region.map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})
print(db)

print("데이터 정보 : \n{}\n".format(db.info()))
print("=================================\n")
print("데이터 크기 : \n{}\n".format(db.shape))
print("=================================\n")
print("데이터 Overview : \n{}\n".format(db.describe()))
print("=================================\n")

print(db.corr(method="pearson"))
X_data = db.iloc[:, 0:6]
y_data = db.iloc[:, -1]
X_data = X_data.values
y_data = y_data.values
print("X 데이터: \n{}\n".format(X_data))
print("y 데이터: \n{}\n".format(y_data))

W = np.random.rand(6, 1)
b = np.random.rand(1)

print("가중치 : {}\n".format(W))
print("바이어스: {}\n".format(b))


def loss_func(x, t):
    y = np.dot(x, W) + b
    return (np.sum((t - y) ** 2)) / (len(x))


def numeric_derivative(f, x):
    delta = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta
        fx1 = f(x)

        x[idx] = tmp_val - delta
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta)

        x[idx] = tmp_val
        it.iternext()
    return grad


def error_val(x, t):
    y = np.dot(x, W) + b
    return (np.sum((t - y) ** 2)) / (len(x))


def predict(x):
    y = np.dot(x, W) + b
    return y


learning_rate = 1e-5

f = lambda x: loss_func(X_data, y_data)
print("초기 에러 : {}, W {}, b = {}\n".format(error_val(X_data, y_data), W, b))

for step in range(10001):
    W -= learning_rate * numeric_derivative(f, W)
    b -= learning_rate * numeric_derivative(f, b)

    if step % 400 == 0:
        print("{}step, 에러 : {}, W = {}, b = {}\n".format(step, error_val(X_data, y_data), W, b))
