import json
import numpy as np
import sys
sys.path.insert(
    0, "/home/carlosvp/dev/school/ia_ii/artificial-intelligence/perceptron")
from perceptron import Perceptron

# ! This class inherits from the Perceptron Class


class Adaline(Perceptron):
    pass


if __name__ == "__main__":
    with open('bulk_data.json', 'r') as json_file:
        data = json.load(json_file)
    args = {'bulk_data': data}
    ada = Adaline(**args)
    ada.process()
    print(ada.weights)
