import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, no_hiddenlayers=2, hl_units=(2, 2)):
        self.hiddenlayers = no_hiddenlayers
        self.hlunits = hl_units
        self.neuralnet = []
        self.final_output = None
        self.loss = []

    def createNeuralNet(self, no_inputs):
        for i in range(self.hiddenlayers):
            hl = [{'weights': np.array([random.gauss(mu=0.0, sigma=1.0) for i in range(no_inputs + 1)])} for j in
                  range(self.hlunits[i])]
            no_inputs = self.hlunits[i]
            self.neuralnet.append(hl)
        output_layer = [{'weights': np.array(
            [random.gauss(mu=0.0, sigma=1.0) for i in range(self.hlunits[self.hiddenlayers - 1] + 1)])}]
        self.neuralnet.append(output_layer)

    def neuralnet_zeroweight(self, no_inputs):
        for i in range(self.hiddenlayers):
            hl = [{'weights': np.array([0.0 for i in range(no_inputs + 1)])} for j in range(self.hlunits[i])]
            no_inputs = self.hlunits[i]
            self.neuralnet.append(hl)
        output_layer = [{'weights': np.array([0.0 for i in range(self.hlunits[self.hiddenlayers - 1] + 1)])}]
        self.neuralnet.append(output_layer)

    def neuronoutput(self, weights, input, layer):
        sum = weights[-1]
        for i in range(len(weights) - 1):
            sum += weights[i] * input[i]
        if layer == len(self.neuralnet) - 1:
            return sum
        return 1 / (1 + np.exp(-sum))

    def forward_propogate(self, input_val):
        for layer in range(len(self.neuralnet)):
            neuron_output = []
            for neuron in self.neuralnet[layer]:
                neuron['output'] = self.neuronoutput(neuron["weights"], input_val, layer)
                neuron['input'] = np.array(input_val)
                neuron_output.append(neuron['output'])
            input_val = neuron_output
        self.final_output = input_val
        return input_val

    def sigmoid_derivative(self, value):
        return value * (1 - value)

    def back_propogate(self, actual_value, lr):
        for i in reversed(range(len(self.neuralnet))):
            layer = self.neuralnet[i]
            if i == len(self.neuralnet) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    error = neuron['output'] - actual_value
                    neuron['cache'] = error
                    neuron['input'] = np.append(neuron['input'], [1])
                    neuron['weights'] -= lr * neuron['cache'] * neuron['input']
            else:
                for j in range(len(layer)):
                    cache = np.zeros(shape=(1,), dtype=float)
                    for neuron in self.neuralnet[i + 1]:
                        cache[0] += neuron['weights'][j] * neuron['cache']
                    layer[j]['cache'] = cache[0]
                    layer[j]['input'] = np.append(layer[j]['input'], [1])
                    layer[j]['weights'] -= lr * layer[j]['cache'] * self.sigmoid_derivative(layer[j]['output']) * \
                                           layer[j]['input']

    def sgd_neuralnet(self, X, y, lr=None, epochs=20):
        for i in range(epochs):
            indexes = random.sample(range(len(X)), len(X))
            loss = 0
            for j in indexes:
                row = X[j]
                self.forward_propogate(row)
                expected = y.iloc[j]
                self.back_propogate(expected, lr(i))
                # loss += 0.5*(self.forward_propogate(row) - expected)**2
            # self.loss.append(loss)

    def predict(self, X_test):
        X = np.array(X_test)
        output = lambda data: self.forward_propogate(data)
        predicted = np.array([output(data)[0] for data in X])
        modified_output = predicted.copy()
        modified_output[predicted > 0.5] = 1
        modified_output[predicted < 0.5] = 0
        return modified_output

    def calculateError(self, actual, predicted):
        return 1 - (np.sum(actual == predicted) / len(actual))


if __name__ == "__main__":
    X_train = pd.read_csv('Data/train.csv', header=None)
    X_test = pd.read_csv('Data/test.csv', header=None)
    y = X_train.iloc[:, 4]
    X_train = X_train.iloc[:, :4]
    y_test = X_test.iloc[:, 4]
    X_test = X_test.iloc[:, :4]
    T = [x for x in range(15)]
    widths = [5, 10, 25, 50, 100]
    gamma = 0.01
    for width in widths:
        print("Width: ", width)
        learning_rate = lambda i: gamma / (1 + (gamma * i) / width)
        nn = NeuralNetwork(no_hiddenlayers=2, hl_units=(width, width))
        nn.createNeuralNet(X_train.shape[1])
        nn.sgd_neuralnet(np.array(X_train), y, learning_rate)
        predicted_train = nn.predict(X_train)
        predicted_test = nn.predict(X_test)
        print("Training Error: ", nn.calculateError(y, predicted_train))
        print("Test Error: ", nn.calculateError(y_test, predicted_test))
        # fig1, ax1 = plt.subplots()
        # ax1.plot(T, nn.loss, color='blue', label="training loss")
        # ax1.set_xlabel('Epochs')
        # ax1.set_ylabel('Loss')
        # ax1.set_title(f"Epochs vs Loss (Width: {width})")
        # ax1.legend()

    print("After Initialising all weights as zero")
    for width in widths:
        print("Width: ", width)
        learning_rate = lambda i: gamma / (1 + (gamma * i) / width)
        nn = NeuralNetwork(no_hiddenlayers=2, hl_units=(width, width))
        nn.neuralnet_zeroweight(X_train.shape[1])
        nn.sgd_neuralnet(np.array(X_train), y, learning_rate)
        predicted_train = nn.predict(X_train)
        predicted_test = nn.predict(X_test)
        print("Training Error: ", nn.calculateError(y, predicted_train))
        print("Test Error: ", nn.calculateError(y_test, predicted_test))
        # fig1, ax1 = plt.subplots()
        # ax1.plot(T, nn.loss, color='blue', label="training loss")
        # ax1.set_xlabel('Epochs')
        # ax1.set_ylabel('Loss')
        # ax1.set_title(f"Epochs vs Loss (Width: {width})")
        # ax1.legend()