import pandas as pd
import numpy as np
import random

class PrimalSVM:
    def __init__(self):
        self.weights = None

    def fit(self,X_train,y_train,lr = None,epochs = 100,C = 0.3):
        #w_0 = np.ones(X_train.shape[1])
        X_train = np.append(np.ones((X_train.shape[0], 1)), X_train, axis=1)
        self.weights = np.zeros(X_train.shape[1])
        for i in range(epochs):
            indexes = random.sample(range(len(X_train)), len(X_train))
            new_lr = lr(i)
            for j in indexes:
               if y_train[j]*np.dot(self.weights,X_train[j])<=1:
                   self.weights = self.weights - new_lr*self.weights + new_lr* C * len(X_train) * y_train[j] * X_train[j]
               else:
                   self.weights = (1-new_lr) * self.weights

    def predict(self,X_test):
        X_test = np.append(np.ones((X_test.shape[0],1)),X_test,axis=1)
        output = lambda data: np.sign(np.dot(self.weights,data))
        predicted = [output(data) for data in X_test]
        return predicted

    def calculateError(self,actual,predicted):
        return 1 - (np.sum(actual == predicted)/len(actual))


if __name__ == "__main__":
    X_train = pd.read_csv('./Bank_data/train.csv',header = None)
    X_test = pd.read_csv('./Bank_data/test.csv',header=None)
    y = X_train.iloc[:,4]
    target = np.array([-1 if i ==0 else 1 for i in y])
    X_train = X_train.iloc[:,:4]
    y_test = X_test.iloc[:,4]
    X_test = X_test.iloc[:,:4]
    actual_y = np.array([-1 if i ==0 else 1 for i in y_test])
    C = [100/873, 500/873, 700/873]
    initial_lr, a = 0.5, 1.5
    print("-- Primal SVM -- part a")
    learning_rate = lambda i : initial_lr / (1 + (initial_lr * i) / a)
    for i in range(len(C)):
        primal_svm = PrimalSVM()
        primal_svm.fit(X_train,target,epochs = 100,lr = learning_rate,C=C[i])
        predicted_y_test = primal_svm.predict(X_test)
        predicted_y_train = primal_svm.predict(X_train)
        print(f"C ->{C[i]} ")
        print("Weight vector: ", primal_svm.weights[1:])
        print("bias: ", primal_svm.weights[0])
        print("Error on training data - ", primal_svm.calculateError(target, predicted_y_train))
        print("Error on test data - ",primal_svm.calculateError(actual_y,predicted_y_test))

    print("-- Primal SVM -- part b")
    learning_rate = lambda i: initial_lr / (1 + i)
    for i in range(len(C)):
        primal_svm = PrimalSVM()
        primal_svm.fit(X_train, target, epochs=100, lr=learning_rate,C=C[i])
        predicted_y_test = primal_svm.predict(X_test)
        predicted_y_train = primal_svm.predict(X_train)
        print(f"C ->{C[i]}")
        print("Weight vector: ", primal_svm.weights[1:])
        print("bias: ", primal_svm.weights[0])
        print("Error on training data - ", primal_svm.calculateError(target, predicted_y_train))
        print("Error on test data - ", primal_svm.calculateError(actual_y, predicted_y_test))
