import json
import numpy as np
import weakref
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
        self.slopes = []

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
            x = np.arange(0, 8)
            for element in self.data:
                error = element['expected'] - self.activation(element['coord'])
                if(error != 0):
                    not_done = True
                    self.error_freq += 1
                    self.adjust(error, element['coord'])
            self.slopes.append(self.weights[0][1])

        self.trained = True


if __name__ == "__main__":
    with open('perceptron/bulk_data.json', 'r') as json_file:
        data = json.load(json_file)

    perceptron = Perceptron(data, eta=0.4, epoch_limit=200)
    perceptron.process()

    for element in data:
        sym = 'ro'
        if(element['expected'] == 1):
            sym = 'bo'
        plt.plot(element['coord'][1], element['coord'][2], sym)

    # The line drawing must be re-done because of the origin of the line
    axes = plt.axis()
    while(perceptron.slopes):
        slope = perceptron.slopes.pop()
        current_line = plt.axline(
            (axes[0]-.2, axes[2]-.2), slope=slope, color="orange")
        plt.pause(0.001)
        wl = weakref.ref(current_line)
        current_line.remove()
        del current_line

    input()
