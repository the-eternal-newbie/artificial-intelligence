# Adaline
The Adaline classifier is closely related to the Ordinary Least Squares (OLS) Linear Regression algorithm; in OLS regression we find the line (or hyperplane) that minimizes the vertical offsets. Or in other words, we define the best-fitting line as the line that minimizes the sum of squared errors (SSE) or mean squared error (MSE) between our target variable (*y*) and our predicted output over all samples *i* in our dataset of size *n*.

![alt text][perceptron]

## Algorithm
The algorithm is pretty simple and its steps are the following:
- As long as the perceptron hasn't learned or hasn't reach the limit of epochs, it will repeat an *epoch*:
  - In each epoch the perceptron will iterate through all the dataset
  - In each iteration the perceptron will check for the error between the desired output and the value given by its activation function:
    - If error is different from zero, the perceptron will must adjust its weights

### Pseudocode
```python
sqr_error = 0
while(sqr_error != desired_error and current_epoch < limit):
    for data in data_set:
        error = data['desired'] - sigmoid(data['coord'])
        adjust_weights()
    limit += 1
```
### Activation Function (Sigmoid)
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