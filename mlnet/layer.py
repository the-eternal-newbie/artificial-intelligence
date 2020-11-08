import numpy as np
from .algo import *

from .adaline import Adaline

class Layer(object):
    def __init__(self, **kwargs):
        self.output = []
        self.neurons = []
        self.algo = kwargs.get('algo')
        self.type = kwargs.get('type', 'hidden')
        for _ in range(kwargs.get('size')):
            neuron = Adaline(**{
                'bulk_data': kwargs.get('data_set'),
                'weights': kwargs.get('weights', None),
                'eta': kwargs.get('eta'),
                'epoch_limit': kwargs.get('epoch_limit'),
                'sqre': kwargs.get('sqre')
            })
            self.neurons.append(neuron)

    def process(self):
        new_input = []
        for neuron in self.neurons:
            new_input.append(bp_forward(neuron)[0])
        print(new_input)