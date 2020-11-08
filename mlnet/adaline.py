import numpy as np
from perceptron import Perceptron


# ! This class inherits from the Perceptron Class
class Adaline(Perceptron):
    def __init__(self, **kwargs):
        self.sqre = kwargs.get('sqre', 1.5)
        super().__init__(**kwargs)

    def sigmoid(self, X):
        y = np.dot(self.weights, np.array(X))
        return 1 / (1 + np.exp(-y))

    def adjust(self, error, X):
        self.weights += self.eta * error * \
            (self.sigmoid(X) * (1 - self.sigmoid(X))) * np.array(X)

    def process(self):
        current_sqre = 0
        while((self.current_epoch <= self.epoch_limit)):
            current_sqre = 0
            self.current_epoch += 1
            for element in self.data:
                error = element['expected'] - self.sigmoid(element['coord'])
                current_sqre += (error * error)
                self.adjust(error, element['coord'])
            self.lines.append(
                [self.weights[0][0], self.weights[0][1], self.weights[0][2]])
            current_sqre /= len(self.data)
            self.error_freq.append(float(current_sqre))
            if(current_sqre <= self.sqre):
                break
