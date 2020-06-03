"""
this file is a work in progress, attempting to implement a deep neural network without the use of outside libraries

"""
import os
import numpy as np
from PIL import Image

def squish(num):
    return 1.0 / (1.0 + np.exp(-num))

def squish_prime(num):
    return squish(num) * (1 - squish(num))

class NeuralNetwork:
    def __init__(self, layers):
        self.number_of_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(layer, 1) for layer in self.layers[1:]]
        self.weights = [np.random.randn(layer_b, layer_a) for layer_a, layer_b in zip(layers[:-1], layers[1:])]

    def forward(self, input):
            for bias, weight in zip(self.biases, self.weights):
                out = squish(np.dot(weight, input) + bias)

    def gradient_descent(self, data, epochs, batch_size, learning_rate, test=None):
        input_size = len(data)
        for epoch in range(epochs):
            batches = [data[slice:slice + batch_size] for slice in range(0, input_size, batch_size)]
            for batch in batches:
                self.update(batch, learning_rate)

    def update(self, batch, learning_rate):
        _biases = [np.zeros(bias.shape) for bias in self.biases]
        _weights = [np.zeros(weight.shape) for weight in self.weights]
        for inputs, output in batch:
            delta_biases, delta_weights = self.backpropagation(inputs, output)
            _biases = [bias + delta for bias, delta in zip(_biases, delta_biases)]
            _weights = [weight + delta for weight, delta in zip(_weights, delta_weights)]
        self.biases = [bias - (learning_rate / len(batch)) * gradient for bias, gradient in zip(self.biases, _biases)]
        self.weights = [weight - (learning_rate / len(batch)) * gradient for weight, gradient in zip(self.weights, _weights)]

    def backpropagation(self, inputs, output):
        _biases = [np.zeros(bias.shape) for bias in self.biases]
        _weights = [np.zeros(weight.shape) for weight in self.weights]
        activation = inputs
        activations = [inputs]
        z_scores = []
        # forward pass
        for bias, weight in zip(self.biases, self.weights):
            confidence = np.dot(weight, activation) + bias
            z_scores.append(confidence)
            activation = squish(confidence)
            activations.append(activation)
        # backwards pass
        y = [0] * 6
        y[int(output) - 1] = 1
        delta = self.cost(activations[-1], output) * squish_prime(z_scores[-1])
        _biases[-1] = delta
        _weights[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.number_of_layers):
            z = z_scores[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * squish_prime(z)
            _biases[-1] = delta
            _weights[-1] = np.dot(delta, activations[-layer-1].transpose())
        return _biases, _weights

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.ffeedforward(x)), y) for (x, y) in test_data]

    def cost(self, oa, y):
        print(oa[0][int(y)-1])
        print(y)
        return (oa[0][int(y)-1] - 1)

if __name__=="__main__":
    test_data = []
    path = "/home/naivoder/ml4py/data"
    for root, dirs, files in os.walk(path):
        for name in files:
            #print(name)
            if name.endswith('PNG'):
                img = Image.open(os.path.join(root, name))
                data = np.asarray(img)
                #print(data.shape)
                _name = name[:-3]
                _name += 'txt'
                answer_file = open(os.path.join(root, _name), 'r')
                answer = answer_file.read().rstrip()
                datapoint = (data, answer)
                test_data.append(datapoint)
    net = NeuralNetwork([1024, 30, 1])
    net.gradient_descent(test_data, 30, 10, 3.0)
