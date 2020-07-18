import numpy as np


# 수치 미분
def numerical_derivative(f, x):
    delta_x = 1e-4
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)


# f(x) = x^2에서 미분계수 f'(3) 구하기

def my_func(x):
    return x ** 2


result = numerical_derivative(my_func, 3)
print("Result = {}\n".format(result))


# f(x) = 3xe^x를 미분한 함수를 f'(x)라 할 때, f'(2) 구하기
def my_func2(x):
    return 3 * x * (np.exp(x))


result2 = numerical_derivative(my_func2, 2)
print("Result2 : {}\n".format(result2))

'''
입력 변수가 하나 이상인 다변수 함수의 경우,
입력 변수는 서로 독립이기 때문에
수치 미분 또한 변수 개수만큼 개별적 계산 필요.

f(x, y) = 2x + 3xy + y^3라면, 입력 변수 x, y는 각각 편미분 필요.
'''


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


'''
1변수 함수 f(x) = x^2, f'(3, 0)
'''


def func1(input):
    x = input[0]
    y = input[1]
    return 2 * x + 3 * x * y + np.power(y, 3)


def one_var_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    print("debug1 : initial input var ={}\n".format(x))
    print("debug2 : initial grad = {}\n".format(grad))
    print("===========================================")

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        print("debug3: idx = {0}, x[idx] = {1}\n".format(idx, x[idx]))

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        print("debug4: grad[idx] = {}\n".format(grad[idx]))
        print("debug5: grad = {}\n".format(grad))
        print("==========================================")

        x[idx] = tmp_val
        it.iternext()
    return grad


input = np.array([1.0, 2.0])
one_var_derivative(func1, input)
