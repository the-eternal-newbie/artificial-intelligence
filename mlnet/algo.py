from .adaline import Adaline

def bp_forward(neuron: Adaline):
    return neuron.process()
    
def bp_backward(neuron: Adaline):
    print('bp backward')

def qp_forward(neuron: Adaline):
    print('qp forward')

def qp_backward(neuron: Adaline):
    print('qp backward')