import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial.distance import pdist, squareform


class DualSVM:
    def __init__(self):
        self.w_star = None
        self.b_star = None
        self.support_vectors = []
        self.alpha = None

    @staticmethod
    def gaussian_kernel(x_i, x_j, gamma):
        return np.exp(-(np.linalg.norm(x_i - x_j, ord=2) ** 2) / gamma)

    @staticmethod
    def inner_optimization_(alpha, X_train, target, kernel, gamma):
        if kernel == "gaussian":
            dist = squareform(pdist(X_train, 'euclidean'))
            gk = np.exp(-np.square(dist) / gamma)
            return 0.5 * np.sum((target[:, None] * target[None, :]) * gk * (alpha[:, None] * alpha[None, :])) - np.sum(
                alpha)
        return 0.5 * np.sum(
            (target[:, None] * target[None, :]) * (X_train @ X_train.T) * (alpha[:, None] * alpha[None, :])) - np.sum(
            alpha)

    @staticmethod
    def getConstraints(C, y_train):
        return {
            'type': 'eq',
            'fun': lambda alpha: np.dot(alpha, y_train)
        }

    def fit(self, X_train, y_train, C=0.3, kernel="linear", gamma=None):
        X_train = np.array(X_train)
        alpha_values = scipy.optimize.minimize(DualSVM.inner_optimization_, x0=np.zeros(len(X_train)),
                                               args=(X_train, y_train, kernel, gamma), method='SLSQP',
                                               constraints=DualSVM.getConstraints(C, y_train), bounds=[(0, C)] * 872)

        self.alpha = alpha_values['x']
        self.w_star = np.sum(
            np.multiply(np.multiply(np.reshape(self.alpha, (-1, 1)), np.reshape(y_train, (-1, 1))), X_train), axis=0)
        self.w_star = self.w_star.reshape(1, -1)
        nonzero_alpha = np.where(self.alpha > 0)[0]
        self.b_star = 0
        x = X_train[nonzero_alpha, :]
        self.support_vectors = x
        y = y_train[nonzero_alpha]
        if kernel == "gaussian":
            for i in nonzero_alpha:
                self.b_star += y_train[i] - DualSVM.gaussian_kernel(self.w_star, X_train[i], gamma)
            self.b_star /= len(nonzero_alpha)
        else:
            self.b_star = np.mean(y - np.dot(self.w_star, x.T)[0])
        self.b_star = np.mean(y - np.dot(self.w_star, x.T)[0])

    def predict(self, X, kernel="linear", gamma=None):
        X = np.array(X)
        if kernel == "gaussian":
            predict = lambda data: np.sign(DualSVM.gaussian_kernel(self.w_star[0], data, gamma) + self.b_star)
            return np.array([predict(data) for data in X])
        return np.sign(np.dot(self.w_star, X.T)[0] + self.b_star)

    def gaussian_predict(self, X, y, X1, gamma=None):
        xvals = X ** 2 @ np.ones_like(X1.T) - 2 * X @ X1.T + np.ones_like(X) @ X1.T ** 2
        xvals = np.exp(-(xvals / gamma))
        k = np.multiply(np.reshape(y, (-1, 1)), xvals)
        y0 = np.sum(np.multiply(np.reshape(self.alpha, (-1, 1)), k), axis=0)
        output = np.array(y0)
        output[output > 0] = 1
        output[output <= 0] = -1
        return output

    def calculateError(self, actual, predicted):
        return 1 - (np.sum(np.equal(actual, predicted)) / len(actual))


if __name__ == "__main__":
    X_train = pd.read_csv('./Bank_data/train.csv', header=None)
    X_test = pd.read_csv('./Bank_data/test.csv', header=None)
    y = X_train.iloc[:, 4]
    target = np.array([-1 if i == 0 else 1 for i in y])
    X_train = X_train.iloc[:, :4]
    y_test = X_test.iloc[:, 4]
    X_test = X_test.iloc[:, :4]
    actual_y = np.array([-1 if i == 0 else 1 for i in y_test])
    C = [100 / 873, 500 / 873, 700 / 873]
    gamma_values = [0.1, 0.5, 1, 5, 100]
    alphavalue = None
    print("-- Dual SVM -- part a")
    for i in range(len(C)):
        dual_svm = DualSVM()
        dual_svm.fit(X_train, target, C[i])
        predicted_y_test = dual_svm.predict(X_test)
        predicted_y_train = dual_svm.predict(X_train)
        print(f"C ->{C[i]} ")
        print("Weight vector: ", dual_svm.w_star)
        print("bias: ", dual_svm.b_star)
        print("Error on training data - ", dual_svm.calculateError(target, predicted_y_train))
        print("Error on test data - ", dual_svm.calculateError(actual_y, predicted_y_test))
    svectors = {}
    print("-- Dual SVM -- part b")
    for i in range(len(C)):
        for g in gamma_values:
            dual_svm = DualSVM()
            dual_svm.fit(X_train, target, C[i], kernel="gaussian", gamma=g)
            predicted_y_train = dual_svm.gaussian_predict(X_train, target, X_train, gamma=g)
            predicted_y_test = dual_svm.gaussian_predict(X_train, target, X_test, gamma=g)
            print(f"C ->{C[i]} ")
            print(f"gamma ->{g} ")
            print("Weight vector: ", dual_svm.w_star)
            print("bias: ", dual_svm.b_star)
            print("No of support vectors: ", len(dual_svm.support_vectors))
            print("Error on training data - ", dual_svm.calculateError(target, predicted_y_train))
            print("Error on test data - ", dual_svm.calculateError(actual_y, predicted_y_test))
            if (C[i] == 500 / 873):
                svectors[g] = dual_svm.support_vectors
    print("--Dual SVM -- part C")
    pairs = [(0.1, 0.5), (0.5, 1), (1, 5), (5, 100)]
    count_common = 0
    for i, j in pairs:
        count_common = 0
        for k in range(len(svectors[i])):
            for l in range(len(svectors[j])):
                if np.array_equal(svectors[i][k], svectors[j][l]):
                    count_common += 1
        print(f"Overlapping support vectors between {i} and {j} are {count_common}")
