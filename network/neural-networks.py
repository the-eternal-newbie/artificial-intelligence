import numpy as np
import matplotlib.pyplot as plt

class Neuron

class Network(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get('bulk_data', None)
        if(self.data == None):
            raise AttributeError
        self.weights = kwargs.get('weights', None)
        self.hidden_layers = kwargs.get('hidden_layers', 1)
        self.layer_neurons = kwargs.get('layer_neurons', 2)
        self.eta = kwargs.get('eta', 0.3)
        self.error_freq = []
        self.current_epoch = 0
        self.epoch_limit = kwargs.get('epoch_limit', 100)


    def sqr_error(self):
        pass

    def propagation(self):
        pass

    def backpropagation(self):
        pass

    def train(self):
        self.propagation()
        self.backpropagation()