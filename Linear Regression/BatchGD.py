import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt


class BatchGradientDescent:
    def __init__(self,lr = 0.01,threshold = 0.0000001 ):
        self.learning_rate = lr
        self.coef = None
        self.intercept = None
        self.threshold = threshold
        self.errors = []

    def fit(self,X,y):
        self.intercept = 0
        self.coef = np.ones(X.shape[1],dtype='float64')
        vector_norm = norm(self.coef,2)
        while vector_norm > self.threshold:   #threshold condition
            y_hat = np.dot(X,self.coef) + self.intercept  #calculating predicted output
            self.errors.append((np.mean(np.square(y - y_hat))))  #calculating error
            derivate_intercept = -2 * np.mean(y - y_hat) #calculating derivative of intercept
            derivative_coefficient = -2 * np.dot((y - y_hat),X)/X.shape[0]   #calculating derivative of weights
            vector_norm = norm(((self.coef - self.learning_rate*derivative_coefficient) - self.coef),2)  #vector norms
            self.coef -= self.learning_rate * derivative_coefficient  #calculating updated coefficient
            self.intercept -= self.learning_rate * derivate_intercept   #calculating updated intercept.

    def predict(self,X_test):
        return np.dot(X_test,self.coef) + self.intercept

    @staticmethod
    def calculateerror(target,predicted):
        return np.mean(np.square(target - predicted))


if __name__ == '__main__':
    learning_rates = [0.3,0.25,0.125,0.05,0.01]
    data = pd.read_csv('./Concrete/train.csv',
                       names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'y'])
    testdata = pd.read_csv('./Concrete/test.csv',
                           names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'y'])
    X = data.iloc[:, :-1]
    y = data['y']
    X_test = data.iloc[:, :-1]
    y_test = data['y']
    errors = {}
    count = 0
    for lr in learning_rates:
        bgd = BatchGradientDescent(lr)
        bgd.fit(X,y)
        predicted = bgd.predict(X_test)
        print("learning rate: ",lr," Weight Vector: ",bgd.coef," Intercept: ",bgd.intercept," Error(Cost): ",BatchGradientDescent.calculateerror(y_test,predicted))
        errors[lr] = bgd.errors
    colors = ['red','maroon','lightcoral','aqua','steelblue','lawngreen']

    for lr in errors:
        iterations = [x+1 for x in range(len(errors[lr]))]
        plt.plot(iterations,errors[lr],color = colors[count],label=lr)
        plt.xscale("log")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Cost/Error")
        plt.title("Cost vs Iterations")
        count += 1
    plt.show()