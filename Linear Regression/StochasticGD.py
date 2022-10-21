import copy
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt


class StochasticGradientDescent:
    def __init__(self,lr = 0.01,error_threshold = 0.05 ):
        self.learning_rate = lr
        self.coef = None
        self.intercept = None
        self.threshold = error_threshold
        self.errors = []
        self.epochs = 10000

    def fit(self,X,y):
        self.intercept = 0
        self.coef = np.ones(X.shape[1])
        labels = self.predict(X)
        error = StochasticGradientDescent.calculateerror(y, labels)
        count = 0
        while error > self.threshold and count<self.epochs:
            for i in range(X.shape[0]):
                index = np.random.randint(0,X.shape[0])
                y_hat = np.dot(X.iloc[index],self.coef) + self.intercept
                #self.errors.append(np.square(y[index] - y_hat))
                derivate_intercept = -2 * (y[index] - y_hat)
                derivative_coefficient = -2 * np.dot((y[index] - y_hat),X.iloc[index])
                self.coef -= self.learning_rate * derivative_coefficient
                self.intercept -= self.learning_rate * derivate_intercept
            labels = self.predict(X)
            error = StochasticGradientDescent.calculateerror(y,labels)
            self.errors.append(error)
            count += 1

    def predict(self,X_test):
        return np.dot(X_test,self.coef) + self.intercept

    @staticmethod
    def calculateerror(target,predicted):
        return np.mean(np.square(target - predicted))


if __name__ == '__main__':
    learning_rates = [0.3, 0.25, 0.125, 0.05, 0.01]
    data = pd.read_csv('./Concrete/train.csv',names = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','y'])
    testdata = pd.read_csv('./Concrete/test.csv',names = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','y'])
    X = data.iloc[:,:-1]
    y = data['y']
    X_test = data.iloc[:,:-1]
    y_test = data['y']
    errors = {}
    count = 0
    for lr in learning_rates:
        sgd = StochasticGradientDescent()
        sgd.fit(X,y)
        predicted = sgd.predict(X_test)
        print("learning rate: ", lr, " Weight Vector: ", sgd.coef, " Intercept: ", sgd.intercept, " Error(Cost): ",StochasticGradientDescent.calculateerror(y_test, predicted))
        errors[lr] = sgd.errors
    colors = ['red', 'maroon', 'lightcoral', 'aqua', 'steelblue', 'lawngreen']
    for lr in errors:
        iterations = [x for x in range(len(errors[lr]))]
        plt.plot(iterations,errors[lr],color = colors[count],label=lr)
        plt.xscale("log")
        plt.legend()
        plt.xlabel("No of Updates")
        plt.ylabel("Cost/Error")
        plt.title("Cost vs Iterations")
        count += 1
    plt.show()