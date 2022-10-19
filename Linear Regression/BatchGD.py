import numpy as np
import pandas as pd
from numpy.linalg import norm
class BatchGradientDescent:
    def __init__(self,lr = 0.01,threshold = 0.0000001 ):
        self.learning_rate = lr
        self.coef = None
        self.intercept = None
        self.threshold = threshold
        self.errors = []

    def fit(self,X,y):
        self.intercept = 0
        self.coef = np.ones(X.shape[1])
        vector_norm = norm(self.coef,2)
        while vector_norm > self.threshold:
            y_hat = np.dot(X,self.coef) + self.intercept
            self.errors.append((np.mean(np.square(y - y_hat))))
            derivate_intercept = -2 * np.mean(y - y_hat)
            derivative_coefficient = -2 * np.dot((y - y_hat),X)/X.shape[0]
            vector_norm = norm(((self.coef - self.learning_rate*derivative_coefficient) - self.coef),2)
            self.coef -= self.learning_rate * derivative_coefficient
            self.intercept -= self.learning_rate * derivate_intercept

    def predict(self,X_test):
        return np.dot(X_test,self.coef) + self.intercept

    @staticmethod
    def calculateerror(target,predicted):
        return np.mean(np.square(target - predicted))

if __name__ == '__main__':
    bgd = BatchGradientDescent()
    data = pd.read_csv('./Concrete/train.csv',names = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','y'])
    testdata = pd.read_csv('./Concrete/test.csv',names = ['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','y'])
    X = data.iloc[:,:-1]
    y = data['y']
    X_test = data.iloc[:,:-1]
    y_test = data['y']
    bgd.fit(X,y)
    predicted = bgd.predict(X_test)
    print(BatchGradientDescent.calculateerror(y_test,predicted))
