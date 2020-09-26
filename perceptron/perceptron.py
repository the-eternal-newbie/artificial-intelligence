import json
import numpy as np
import weakref
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# * Notes: Depending on the distribution and amount of data and its linear relationship,
# * the convergence of the algorithm can be faster or slower;
# * the weights are really important to adjust the division line
class Perceptron(object):
    def __init__(self, **kwargs):
        self.eta = kwargs.get('eta', 0.3)
        self.data = kwargs.get('bulk_data', None)
        if(self.data == None):
            raise AttributeError
        self.error_freq = []
        self.current_epoch = 0
        self.epoch_limit = kwargs.get('epoch_limit', 100)
        if(kwargs.get('weights', None)):
            self.weights = np.array([weights])
        else:
            self.weights = np.random.rand(1, 3)
        self.lines = []

    #       Perceptron function Pω(x^j):
    # !         ωx
    def activation(self, X):
        # np.dot() function is a function to calculate the dot product between two or more vectors
        if(np.dot(self.weights, np.array(X)) >= 0):
            return(1)
        return(0)

    # Adjust the weights depending on the error given by
    # the difference between the value given by Pω(x^j) and y^j:
    # !     Δω = η * ε * X
    def adjust(self, error, X):
        self.weights += self.eta * error * np.array(X)

    def process(self):
        not_done = True
        # Stop criteria: keep unless the process doesn't find errors (error == 0)
        # or the epoch limit is reached
        while(not_done and (self.current_epoch <= self.epoch_limit)):
            self.current_epoch += 1
            error_freq = 0
            not_done = False
            for element in self.data:
                error = element['expected'] - self.activation(element['coord'])
                if(error != 0):
                    not_done = True
                    error_freq += 1
                    self.adjust(error, element['coord'])
                    self.lines.append(
                        [self.weights[0][0], self.weights[0][1], self.weights[0][2]])
            self.error_freq.append(error_freq)
