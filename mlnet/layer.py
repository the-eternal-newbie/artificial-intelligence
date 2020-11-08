import numpy as np

from adaline import Adaline

class Layer(object):
    def __init__(self, **kwargs):
        self.output = []
        self.neurons = []
        self.algo = kwargs.get('algo')
        for _ in range(kwargs.get('size')):
            neuron = Adaline(**{
                'bulk_data': kwargs.get('data_set'),
                'weights': kwargs.get('weights'),
                'eta': kwargs.get('eta'),
                'epoch_limit': kwargs.get('epoch_limit'),
                'sqre': kwargs.get('sqr')
            })
            self.neurons.append(neuron)

    def process(self):
        if(self.algo == 'backprop'):
            print('backprop')
        elif(self.algo == 'quickprop'):
            print('quickprop')