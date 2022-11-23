import pandas as pd
import numpy as np


class kernel_perceptron:
    def __init(self):
        self.mistake_counter = None
        self.support_vectors = []

    @staticmethod
    def gaussian_kernel(x_i, x_j, gamma):
        return np.exp(-(np.linalg.norm(x_i - x_j, ord=2) ** 2) / gamma)

    def fit(self, X_train, y_train, gamma=None, epochs=10):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        xvals = X_train ** 2 @ np.ones_like(X_train.T) - 2 * X_train @ X_train.T + np.ones_like(
            X_train) @ X_train.T ** 2
        grammat = np.exp(-(xvals / gamma))
        self.mistake_counter = np.zeros(X_train.shape[0])
        for t in range(epochs):
            for j in range(X_train.shape[0]):
                sum = 0
                for i in range(X_train.shape[0]):
                    sum += self.mistake_counter[i] * y_train[i] * kernel_perceptron.gaussian_kernel(X_train[i],
                                                                                                    X_train[j], gamma)
                y_hat = np.sign(sum)
                if y_hat != y_train[j]:
                    self.mistake_counter[j] += 1
        svectors = self.mistake_counter > 0
        self.mistake_counter = self.mistake_counter[svectors]
        self.support_vectors = X_train[svectors]
        self.output = y_train[svectors]

    def predict(self,X_test, gamma=None):
        predicted = []
        X_test = np.array(X_test)
        for j in range(X_test.shape[0]):
            sum = 0
            for i in range(len(self.support_vectors)):
                sum += self.mistake_counter[i] * self.output[i] * kernel_perceptron.gaussian_kernel(
                    self.support_vectors[i], X_test[j], gamma)
            predicted.append(np.sign(sum))
        return predicted

    def calculateError(self, actual, predicted):
        return 1 - (np.sum(actual == predicted) / len(actual))


if __name__ == "__main__":
    X_train = pd.read_csv('./Bank_data/train.csv', header=None)
    X_test = pd.read_csv('./Bank_data/test.csv', header=None)
    y = X_train.iloc[:, 4]
    target = np.array([-1 if i == 0 else 1 for i in y])
    X_train = X_train.iloc[:, :4]
    y_test = X_test.iloc[:, 4]
    X_test = X_test.iloc[:, :4]
    actual_y = np.array([-1 if i == 0 else 1 for i in y_test])
    gamma = [0.1, 0.5, 1, 5, 100]
    for g in gamma:
        print("gamma: ", g)
        kp = kernel_perceptron()
        kp.fit(X_train, target, gamma=g, epochs=3)
        predicted_y_test = kp.predict(X_test, gamma=g)
        predicted_y_train = kp.predict(X_train, gamma=g)
        print("Error on training data - ", kp.calculateError(target, predicted_y_train))
        print("Error on test data - ", kp.calculateError(actual_y, predicted_y_test))
