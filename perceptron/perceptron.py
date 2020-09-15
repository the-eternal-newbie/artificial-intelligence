import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, bulk_data, eta=0.3, epoch_limit=100):
        self.eta = eta
        self.error_freq = 0
        self.trained = False
        self.data = bulk_data
        self.current_epoch = 0
        self.epoch_limit = epoch_limit
        self.weights = np.random.rand(1, 3)

    def activation(self, X):
        if(np.dot(self.weights, np.array(X)) >= 0):
            return(1)
        return(0)

    def adjust(self, error, X):
        self.weights += self.eta * error * np.array(X)

    def process(self):
        not_done = True
        while(not_done and (self.current_epoch <= self.epoch_limit)):
            self.current_epoch += 1
            not_done = False
            for element in self.data:
                error = element['expected'] - self.activation(element['coord'])
                if(error != 0):
                    not_done = True
                    self.error_freq += 1
                    self.adjust(error, element['coord'])
        self.trained = True
        # n_samples = X.shape[0]
        # n_features = X.shape[1]

        # # Add 1 for the bias term
        # self.weights = np.zeros((n_features+1,))

        # # Add column of 1s
        # X = np.concatenate([X, np.ones((n_samples, 1))], axis=1)

        # for i in range(n_iter):
        #     for j in range(n_samples):
        #         if y[j]*np.dot(self.weights, X[j, :]) <= 0:
        #             self.weights += y[j]*X[j, :]


def load_data():
    URL_ = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header=None)
    print(data)

    # make the dataset linearly separable
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1] == 'Iris-setosa', 0, 1)
    data = np.asmatrix(data, dtype='float64')
    return data


if __name__ == "__main__":
    with open('perceptron/bulk_data.json', 'r') as json_file:
        data = json.load(json_file)

    for element in data:
        if(element['expected'] == 1):
            plt.plot(element['coord'][1], element['coord'][2], 'bo')
        else:
            plt.plot(element['coord'][1], element['coord'][2], 'ro')
    perceptron = Perceptron(data, eta=0.5, epoch_limit=100)
    y = np.array
    print(perceptron.weights)
    plt.show()
