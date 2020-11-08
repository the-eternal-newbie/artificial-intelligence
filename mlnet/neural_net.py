import numpy as np
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get('bulk_data', None)
        if(self.data == None):
            raise AttributeError
        self.weights = kwargs.get('weights', None)

        self.error_freq = []
        self.eta = kwargs.get('eta', 0.3)
        self.sqre = kwargs.get('sqre', 0.1)

        self.current_epoch = 0
        self.epoch_limit = kwargs.get('epoch_limit', 100)
 
        self.algo = kwargs.get('algo', 'backprop')
        self.hidden_layers = kwargs.get('hidden_layers', 1)
        self.layer_neurons = kwargs.get('layer_neurons', 2)

    def sqr_error(self):
        pass

    def learn(self):
        if(self.algo == 'backprop'):
            pass
        elif(self.algo == 'quickprop'):
            pass