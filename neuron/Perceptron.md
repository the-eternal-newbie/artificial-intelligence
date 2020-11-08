# Perceptron
The idea of the perceptron is analogous to the operating principle of the basic processing unit of the brain: the neuron. A neuron is made up of many input signals carried by dendrites, the cell body, and one output signal carried by the axon. Then, the neuron fires an action signal when the cell reaches a particular threshold; such action may or may not happen. 

Similarly, the perceptron has many inputs (often called features) that are fed by a linear unit that produces a binary output. Therefore, perceptrons can be applied to solve binary classification problems in which the sample must be identified as belonging to one of two predefined classes.

![alt text][perceptron]

## Algorithm
The algorithm is pretty simple and its steps are the following:
- As long as the perceptron hasn't learned or hasn't reach the limit of epochs, it will repeat an *epoch*:
  - In each epoch the perceptron will iterate through all the dataset
  - In each iteration the perceptron will check for the error between the desired output and the value given by its activation function:
    - If error is different from zero, the perceptron will must adjust its weights

### Pseudocode
```python
learned = False
while(learned != True and current_epoch < limit):
    for data in data_set:
        error = data['desired'] - activation(data['coord'])
        if(error != 0):
            adjust_weights()
        else:
            learned = True
    limit += 1
```
### Activation Function
Due the perceptron is a binary classifier, we can define its transfer (or activation) function as follows:

![alt text][activation]

### Weights adjustment
Weight adjustment is made when an error is found in the weight calculation. This error is given by the difference between the desired output and the transfer function.

#### Error
![alt text][error]

#### Adjustment
![alt text][adjust]

[perceptron]: img/model.jpeg
[adjust]: img/adjust.png
[error]: img/error.png
[activation]: img/activation.png