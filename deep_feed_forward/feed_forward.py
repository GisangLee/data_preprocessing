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


class Logistic:
    def __init__(self, gate_name, xdata, tdata):
        self.gate_name = gate_name

        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        # Second Hidden Layer : 6
        self.__W2 = np.random.rand(2, 6)
        self.__b2 = np.random.rand(6)

        self.__W3 = np.random.rand(6, 1)
        self.__b3 = np.random.rand(1)

        self.learning_rate = 1e-2
        print("{} object is created".format(self.gate_name))

    def feed_forward(self):
        delta = 1e-7
        z2 = np.dot(self.__xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.__W3) + self.__b3
        y = a3 = sigmoid(z3)

        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log((1 - y) + delta))

    def loss_val(self):
        delta = 1e-7

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.__W3) + self.__b3
        y = a3 = sigmoid(z3)

        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log((1 - y) + delta))

    def train(self):
        f = lambda x: self.feed_forward()
        print("초기 손실 : {}".format(self.loss_val()))

        for step in range(10001):
            self.__W2 -= self.learning_rate * numerical_derivative(f, self.__W2)
            self.__b2 -= self.learning_rate * numerical_derivative(f, self.__b2)
            self.__W3 -= self.learning_rate * numerical_derivative(f, self.__W3)
            self.__b3 -= self.learning_rate * numerical_derivative(f, self.__b3)

            if step % 400 == 0:
                print("{}step 손실 : {}\n".format(step, self.loss_val()))

    def predict(self, input_data):
        z2 = np.dot(input_data, self.__W2) + self.__b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.__W3) + self.__b3
        y = a3 = sigmoid(z3)

        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result


# AND
and_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_tdata = np.array([0, 0, 0, 1])
and_obj = Logistic("AND_GATE", and_xdata, and_tdata)

and_obj.train()
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    print("{} = {}\n".format(data, and_obj.predict(data)[-1], end=''))

# OR
or_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_tdata = np.array([0, 1, 1, 1])

or_obj = Logistic("OR_GATE", or_xdata, or_tdata)
or_obj.train()

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    print("{} = {}]\n".format(data, or_obj.predict(data)[-1], end=''))

# NAND
nand_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_tdata = np.array([1, 1, 1, 0])

nand_obj = Logistic("NAND_GATE", nand_xdata, nand_tdata)
nand_obj.train()

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    print("{} = {}\n".format(data, nand_obj.predict(data)[-1], end=''))

# XOR
xor_xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_tdata = np.array([0, 1, 1, 0])
xor_obj = Logistic("XOR_GATE", xor_xdata, xor_tdata)
xor_obj.train()

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for data in test_data:
    print("{} = {}\n".format(data, xor_obj.predict(data)[-1], end=''))
