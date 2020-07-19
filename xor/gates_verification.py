import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numerical_derivative(f, x):
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


class LogicGate:
    def __init__(self, gate_name, x_data, t_data):
        self.gate_name = gate_name

        self.__xdata = x_data.reshape(4, 2)
        self.__tdata = t_data.reshape(4, 1)

        self.__W = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        self.learning_rate = 1e-2

    # 손실 함수
    def __loss_func(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log((1 - y) + delta))

    # 손실 함수 값 계산
    def error_val(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log((1 - y) + delta))

    # 수치미분 -> 손실함수 값 최소화
    def train(self):
        f = lambda x: self.__loss_func()
        print("초기 값: {}\n".format(self.error_val()))
        for step in range(8001):
            self.__W -= self.learning_rate * numerical_derivative(f, self.__W)
            self.__b -= self.learning_rate * numerical_derivative(f, self.__b)
            if step % 20 == 0:
                print("{} step, error : {}\n".format(step, self.error_val()))

    def predict(self, input_data):
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)

        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result


# AND
and_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_tdata = np.array([0, 0, 0, 1])

AND_obj = LogicGate("AND_GATE", and_xdata, and_tdata)
AND_obj.train()

print("AND GATE")
test_data = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print("입력 값: {} = 논리 값 = {}\n".format(input_data, logical_val))

# OR
or_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_tdata = np.array([0, 1, 1, 1])

OR_obj = LogicGate("OR_GATE", or_xdata, or_tdata)
OR_obj.train()

print("OR GATE")
test_data = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print("입력 값 = {}, 논리 값 = {}\n".format(input_data, logical_val))

# NAND
nand_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_tdata = np.array([1, 1, 1, 0])

NAND_obj = LogicGate("NAND_GATE", nand_xdata, nand_tdata)
NAND_obj.train()

test_data = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])

print("NAND_GATE")
for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print("입력 값: {}, 논리 값: {}\n".format(input_data, logical_val))


# XOR
xor_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_tdata = np.array([0, 1, 1, 0])

XOR_obj = LogicGate("XOR_GATE", xor_xdata, xor_tdata)
XOR_obj.train()

test_data = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])

print("XOR_GATE")
for input_data in test_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(input_data)
    print("입력 값: {}, 논리 값: {}\n".format(input_data, logical_val))

'''
XOR를 해결하기 위한 방법
    -> NAND , OR, AND를 조합한다.
'''

input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
AND = []
OR = []

new_input_data = []
final_data = []

for index in range(len(input_data)):
    AND = NAND_obj.predict(input_data[index])
    OR = OR_obj.predict(input_data[index])
    print("{}\n".format(AND))
    print("{}\n".format(OR))

    new_input_data.append(AND[-1])
    new_input_data.append(OR[-1])
    print("NEW_INPUT : {}\n".format(new_input_data))

    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))
    final_data.append(logical_val)
    print("FINAL : {}\n".format(final_data))
    new_input_data = []

for index in range(len(input_data)):
    print(input_data[index], ' = ', final_data[index], end='')
    print("\n")


