import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork_pytorch(nn.Module):
    def __init__(self, input_size, no_hiddenlayers, hidden_size, output_size, activation_fn="tanh"):
        super().__init__()
        if activation_fn == "relu":
            self.hidden_layer = nn.ModuleList([])
            self.hidden_layer.append(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()))
            for i in range(no_hiddenlayers - 1):
                self.hidden_layer.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
        else:
            self.hidden_layer = nn.ModuleList([])
            self.hidden_layer.append(nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh()))
            for i in range(no_hiddenlayers - 1):
                self.hidden_layer.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()))
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x_train):
        for layer in self.hidden_layer:
            x_train = layer(x_train)
        output = self.output_layer(x_train)
        return output

    def train_model(self, loss_func, optimiser, epochs=20):
        loss_hist = [0] * epochs
        accurate = [0] * epochs
        for i in range(epochs):
            self.train()
            for x, y in train_dataset:
                pred = self(x)
                pred = pred.reshape(pred.shape[0], )
                loss = loss_func(pred, y)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                loss_hist[i] += loss.item() * y.size(0)
                is_correct = (torch.argmax(pred) == y).float()
                accurate[i] += is_correct.mean()
            loss_hist[i] /= len(train_dataset.dataset)
            accurate[i] /= len(train_dataset.dataset)
            # print(f"error epoch {i}: {loss_hist[i]}")

    def predict(self, X_test):
        return self(X_test)

    def calculateErrors(self, actual, predicted):
        predicted[predicted > 0.5] = 1
        predicted[predicted < 0.5] = 0
        predicted = predicted.reshape(len(predicted), )
        correct = torch.sum(predicted == actual)
        return 1 - correct.item() / len(actual)

    @staticmethod
    def xavier_initialize(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    @staticmethod
    def he_initialize(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)


if __name__ == "__main__":
    X_train = pd.read_csv('Data/train.csv', header=None)
    X_test = pd.read_csv('Data/test.csv', header=None)
    y = np.array(X_train.iloc[:, 4])
    X_train = np.array(X_train.iloc[:, :4])
    y_test = np.array(X_test.iloc[:, 4])
    X_test = np.array(X_test.iloc[:, :4])

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    dataset = TensorDataset(X_train, y_train)

    train_dataset = DataLoader(dataset, batch_size=10, shuffle=True)

    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    print("----RELU----")
    for depth in depths:
        for width in widths:
            print(f"Depth: {depth} Width: {width}")
            input_size = X_train.shape[1]
            hidden_size = width
            output_size = 1
            no_hidden_layers = depth
            nnmodel = NeuralNetwork_pytorch(input_size, no_hidden_layers, hidden_size, output_size, "relu")
            nnmodel.apply(NeuralNetwork_pytorch.he_initialize)
            loss_func = nn.MSELoss()
            optimiser = torch.optim.Adam(nnmodel.parameters(), lr=0.001)
            nnmodel.train_model(loss_func, optimiser)
            predicted_train = nnmodel.predict(X_train)
            predicted_test = nnmodel.predict(X_test)
            print("Training error: ", nnmodel.calculateErrors(y_train, predicted_train))
            print("Test error: ", nnmodel.calculateErrors(y_test, predicted_test))

    print("----TANH----")
    for depth in depths:
        for width in widths:
            print(f"Depth: {depth} Width: {width}")
            input_size = X_train.shape[1]
            hidden_size = width
            output_size = 1
            no_hidden_layers = depth
            nnmodel = NeuralNetwork_pytorch(input_size, no_hidden_layers, hidden_size, output_size)
            nnmodel.apply(NeuralNetwork_pytorch.xavier_initialize)
            loss_func = nn.MSELoss()
            optimiser = torch.optim.Adam(nnmodel.parameters(), lr=0.001)
            nnmodel.train_model(loss_func, optimiser)
            predicted_train = nnmodel.predict(X_train)
            predicted_test = nnmodel.predict(X_test)
            print("Training error: ", nnmodel.calculateErrors(y_train, predicted_train))
            print("Test error: ", nnmodel.calculateErrors(y_test, predicted_test))