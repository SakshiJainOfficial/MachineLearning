import numpy as np
import pandas as pd

def analytical(X,y):
    X = np.append(np.ones((X.shape[0],1)),X,axis=1)
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

if __name__ == "__main__":
    data = pd.read_csv('./Concrete/train.csv',
                       names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'y'])
    testdata = pd.read_csv('./Concrete/test.csv',
                           names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'y'])
    X = data.iloc[:, :-1]
    y = data['y']
    X_test = data.iloc[:, :-1]
    y_test = data['y']
    theta = analytical(X,y)
    print("weights from analytical form:", theta)
    print("intercept: ", theta[0])
    print("weights: ", theta[1:])