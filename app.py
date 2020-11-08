import sys
import json
sys.path.insert(0, "./mlnet")

from mlnet.neural_net import Network

if __name__ == "__main__":
    with open('bulk_data.json', 'r+') as file:
        data_set = json.load(file)
    net = Network(**{
        'bulk_data': data_set,
        'algo': 'backprop',
        'hidden_layers': 2,
        'layer_neurons': 3,
        'num_class': 2
    })
    net.learn()