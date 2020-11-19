import numpy as np
from .perceptron import Perceptron
# from perceptron import Perceptron


# ! This class inherits from the Perceptron Class
class Adaline(Perceptron):
    def __init__(self, **kwargs):
        self.sqre = kwargs.get('sqre', 1.5)
        self.net = []
        super().__init__(**kwargs)

    def net_value(self):
        for element in self.data:
            self.net.append({
                'data': self.sigmoid(np.dot(self.weights, element['data']), net=True)[0],
                'desired': element['desired'],
                'net': np.dot(self.weights, element['data'])
            })
        return self.net

    def sigmoid(self, X, net=False):
        y = X
        if not(net):
            y = np.dot(self.weights, np.array(X))
        return 1 / (1 + np.exp(-y))

    def delta_sig(self, X):
        return(self.sigmoid(X) * (1 - self.sigmoid(X)))

    def sensibility(self, values):
        s = 0
        for data in values:
            s += self.delta_sig(data['n']) * (data['desired'] - data['data'])
        return(-2 * s)

    def adjust(self, error, X):
        self.weights += self.eta * error * self.delta_sig(X) * np.array(X)

    def process(self):
        current_sqre = 0
        while((self.current_epoch <= self.epoch_limit)):
            current_sqre = 0
            self.current_epoch += 1
            for element in self.data:
                error = element['desired'] - self.sigmoid(element['data'])
                current_sqre += (error * error)
                self.adjust(error, element['data'])
            self.lines.append(
                [self.weights[0][0], self.weights[0][1], self.weights[0][2]])
            current_sqre /= len(self.data)
            self.error_freq.append(float(current_sqre))
            if(current_sqre <= self.sqre):
                break
