import numpy as np
import pandas as pd
import random
import copy
class Perceptron:
    def __init__(self):
        self.learning_rate = None
        self.weights = None
        self.epochs = None
        self.votes = None

    def fit_standardperceptron(self,X,y,lr=0.01,epochs=100):
        self.learning_rate = lr
        self.epochs = epochs
        X = np.append(np.ones((X.shape[0],1)),X,axis=1)
        self.weights = np.zeros(X.shape[1])
        for i in range(self.epochs):
            indexes = random.sample(range(len(X)),len(X))
            for j in indexes:
                if y[j]*np.dot(self.weights,X[j])<=0:
                    self.weights += self.learning_rate * (y[j]*X[j])

    def fit_votedperceptron(self,X,y,lr = 0.01,epochs = 100):
        self.learning_rate = lr
        self.epochs = epochs
        m = 0
        C = [0]
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        self.weights = [np.zeros(X.shape[1])]
        for i in range(self.epochs):
            #indexes = random.sample(range(len(X)), len(X))
            for j in range(X.shape[0]):
                if y[j]*np.dot(self.weights[m],X[j])<=0:
                    new_weights = self.weights[m] + self.learning_rate * (y[j]*X[j])
                    self.weights.append(new_weights)
                    m += 1
                    C.append(1)
                else:
                    C[m] += 1
        self.votes = np.array(list(zip(self.weights,C)),dtype=object)

    def fit_averageperceptron(self,X,y,lr=0.01,epochs=100):
        self.learning_rate = lr
        self.epochs = epochs
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        self.weights = np.zeros(X.shape[1])
        w = np.zeros(X.shape[1])
        for i in range(self.epochs):
            #indexes = random.sample(range(len(X)), len(X))
            for j in range(X.shape[0]):
                if y[j]*np.dot(w,X[j])<=0:
                    w += self.learning_rate * (y[j]*X[j])
                self.weights += w


    def predict(self,X_test):
        X_test = np.append(np.ones((X_test.shape[0],1)),X_test,axis=1)
        output = lambda data: np.sign(np.dot(self.weights,data))
        predicted = [output(data) for data in X_test]
        return predicted

    def predict_votedperceptron(self,X_test):
        X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
        predicted = []
        for i in range(len(X_test)):
            sum_weights = 0
            for w,c in self.votes:
                sum_weights += c * np.sign(np.dot(w,X_test[i]))
            predicted.append(np.sign(sum_weights))
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

    print("-- STANDARD PERCEPTRON --")
    per = Perceptron()
    per.fit_standardperceptron(X_train,target,epochs = 10)
    predicted_y = per.predict(X_test)
    print("learned weight vector - ",per.weights)
    print("Error on test data - ",per.calculateError(actual_y,predicted_y))

    print("-- VOTED PERCEPTRON --")
    voted = Perceptron()
    voted.fit_votedperceptron(X_train,target,epochs=10)
    predicted_voted = voted.predict_votedperceptron(X_test)
    print("weight vectors \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t Counts")
    #print("learned weight vector and their counts - ", voted.votes)
    for vect in voted.votes:
        print(f"{vect[0]}\t\t\t\t\t\t{vect[1]}")
    print("Error on test data - ", voted.calculateError(actual_y,predicted_voted))

    print("-- AVERAGE PERCEPTRON --")
    average = Perceptron()
    average.fit_averageperceptron(X_train,target,epochs=10)
    predicted_average = average.predict(X_test)
    print("learned weight vector for - ",average.weights)
    print("Error on test data - ",average.calculateError(actual_y,predicted_average))

