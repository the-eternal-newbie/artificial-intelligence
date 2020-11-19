import numpy as np
from .algo import *

from .adaline import Adaline

class Layer(object):
    def __init__(self, **kwargs):
        self.output = []
        self.neurons = []
        self.sensitivity = 0
        self.eta = kwargs.get('eta')
        self.sqre = kwargs.get('sqre')
        self.algo = kwargs.get('algo')
        self.size = kwargs.get('size')
        self.type = kwargs.get('type', 'hidden')
        self.epoch_limit = kwargs.get('epoch_limit')

    def forward(self, data_set):
        neuron_output = {}
        i = 0
        for _ in range(self.size):
            neuron = Adaline(**{
                'bulk_data': data_set,
                'eta': self.eta,
                'epoch_limit': self.epoch_limit,
                'sqre': self.sqre
            })
            neuron_output[str(i)] = []
            self.neurons.append(neuron)
            i += 1

        # Forward propagation
        i = 0
        for neuron in self.neurons:
            neuron.process()
            neuron_output[str(i)].extend(neuron.net_value())
            i += 1

        for i in range(len(neuron_output['0'])):
            data = [-1]
            desired = neuron_output['0'][i]['desired']
            for key in neuron_output:
                data.append(neuron_output[key][i]['data'])
            self.output.append({'data': data, 'desired': desired})

    def backward(self, sensibility):
        for neuron in self.neurons:
            neuron.process()
            neuron_output[str(i)].extend(neuron.net_value())
            i += 1
