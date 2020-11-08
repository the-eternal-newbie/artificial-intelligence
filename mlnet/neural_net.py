import json
import numpy as np
import matplotlib.pyplot as plt

from layer import Layer

class Network(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get('bulk_data', None)
        if(self.data == None):
            raise AttributeError

        self.error_freq = []
        self.eta = kwargs.get('eta', 0.3)
        self.sqre = kwargs.get('sqre', 0.1)

        self.current_epoch = 0
        self.epoch_limit = kwargs.get('epoch_limit', 100)
 
        self.layers = []
        for _ in range(kwargs.get('hidden_layers', 1)):
            layer = Layer(**{
                'algo': kwargs.get('algo', 'backprop'),
                'data_set': self.data,
                'eta': self.eta,
                'epoch_limit': self.epoch_limit,
                'sqre': self.sqre,
                'size': kwargs.get('layer_neurons', 2)
            })
            self.layers.append(layer)

    def learn(self):
        for layer in self.layers:
            layer.process()

if __name__ == "__main__":
    with open('bulk_data.json', 'r+') as file:
        data_set = json.load(file)
    net = Network(**{
        'bulk_data': data_set,
        'algo': 'backprop',
        'hidden_layers': 2,
        'layer_neurons': 3,
    })
    net.learn()