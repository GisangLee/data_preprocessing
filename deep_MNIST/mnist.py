import numpy as np
from tensorflow.python.client import device_lib

training_data = np.loadtxt('../data_sets/mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('../data_sets/mnist_test.csv', delimiter=',', dtype=np.float32)
print("training : {}, test : {}\n".format(training_data.shape, test_data.shape))


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


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.__W2 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.__b2 = np.random.rand(self.hidden_nodes)

        self.__W3 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.__b3 = np.random.rand(self.output_nodes)

        self.learning_rate = 1e-4

    def feed_forward(self):
        delta = 1e-7
        z1 = np.dot(self.input_data, self.__W2) + self.__b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1, self.__W3) + self.__b3
        y = sigmoid(z2)

        return -np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))

    def loss_val(self):
        pass

    def train(self, training_data):
        self.target_data = np.zeros(self.output_nodes) + 0.01
        self.target_data[int(training_data[0])] = 0.99

        self.input_data = (training_data[1:] / 255.0 * 0.99) + 0.01

        f = lambda x: self.feed_forward()

        self.__W2 -= self.learning_rate * numerical_derivative(f, self.__W2)
        self.__b2 -= self.learning_rate * numerical_derivative(f, self.__b2)

        self.__W3 -= self.learning_rate * numerical_derivative(f, self.__W3)
        self.__b3 -= self.learning_rate * numerical_derivative(f, self.__b3)

    def predict(self, input_data):
        z1 = np.dot(input_data, self.__W2) + self.__b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1, self.__W3) + self.__b3
        y = sigmoid(z2)

        predicted_num = np.argmax(y)
        return predicted_num

    def accuracy(self, test_data):
        match_list = []
        not_match_list = []

        for index in range(len(test_data)):
            label = int(test_data[index, 0])

            data = (test_data[index, 1:] / 255.0 * 0.09) + 0.01
            predicted_num = self.predict(data)

            if label == predicted_num:
                match_list.append(index)
            else:
                not_match_list.append(index)
        print("현재 정확도 : {}%\n".format(100 * (len(match_list) / len(test_data))))
        return match_list, not_match_list


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

for step in range(30001):
    index = np.random.rand(0, len(training_data) - 1)
    nn.train(training_data[index])

    if step % 400 == 0:
        print("{} step, 손실 : {}\n".format(step, nn.loss_val()))
