import json
import numpy as np
import matplotlib.pyplot as plt

from .layer import Layer


class Network(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get('bulk_data', None)
        if(self.data == None):
            raise AttributeError

        self.error_freq = []
        self.eta = kwargs.get('eta', 0.3)
        self.sqre = kwargs.get('sqre', 0.1)
        self.num_class = kwargs.get('num_class', 2)

        self.current_epoch = 0
        self.epoch_limit = kwargs.get('epoch_limit', 100)

        self.layers = []
        for _ in range(kwargs.get('hidden_layers', 1)):
            layer = Layer(**{
                'algo': kwargs.get('algo', 'backprop'),
                'eta': self.eta,
                'epoch_limit': self.epoch_limit,
                'sqre': self.sqre,
                'size': kwargs.get('layer_neurons', 2)
            })
            self.layers.append(layer)

        self.layers.append(Layer(**{
            'type': 'output',
            'algo': kwargs.get('algo', 'backprop'),
            'eta': self.eta,
            'epoch_limit': self.epoch_limit,
            'sqre': self.sqre,
            'size': self.num_class
        }))

    def learn(self):
        output = self.data
        for layer in self.layers:
            layer.process(output)
            output = layer.output
        print(output)