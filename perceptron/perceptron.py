import json
import numpy as np
import weakref
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# * Notes: Depending on the distribution and amount of data and its linear relationship,
# * the convergence of the algorithm can be faster or slower;
# * the weights are really importan to adjust the division line


class Perceptron(object):
    def __init__(self, bulk_data, eta=0.3, epoch_limit=100, weights=None):
        self.eta = eta
        self.data = bulk_data
        self.error_freq = []
        self.current_epoch = 0
        self.epoch_limit = epoch_limit
        if(weights):
            self.weights = np.array([weights])
        else:
            self.weights = np.random.rand(1, 3)
        self.lines = []

    #       Perceptron function Pω(x^j):
    # !         ωx
    def activation(self, X):
        # np.dot() function is a function to calculat the dot product between two or more vectors
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


if __name__ == "__main__":
    with open('test.json', 'r') as json_file:
        data = json.load(json_file)

    perceptron = Perceptron(data, eta=0.1, epoch_limit=500)
    perceptron.process()

    # for element in data:
    #     sym = 'ro'
    #     if(element['expected'] == 1):
    #         sym = 'bo'
    #     plt.plot(element['coord'][1], element['coord'][2], sym)

    # The line drawing must be re-done because of the origin of the line
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x = np.arange(perceptron.current_epoch)
    plt.bar(x, height=perceptron.error_freq)
    print(perceptron.error_freq)
    # plt.xticks(x,)
    # x = np.linspace(-5, 5, 100)
    # y = y = (perceptron.weights[0][0] -
    #          (perceptron.weights[0][1]*x))/perceptron.weights[0][2]
    # plt.plot(x, y)
    plt.show()
    # axes = plt.axis()
    # while(perceptron.slopes):
    #     slope = perceptron.slopes.pop()
    #     current_line = plt.axline(
    #         (axes[0]-.2, axes[2]-.2), slope=slope, color="orange")
    #     plt.pause(0.001)
    #     wl = weakref.ref(current_line)
    #     current_line.remove()
    #     del current_line

    input()
