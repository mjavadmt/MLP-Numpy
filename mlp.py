# Q2_graded
# Do not change the above line.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
np.random.seed(0)
# Dataset
X,Y = datasets.make_circles(n_samples=576, shuffle=True, noise=0.25, random_state=None, factor=0.4)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='plasma')
plt.grid()
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('Data')
plt.show()

# Q2_graded
# Do not change the above line.

X_train = X.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # implement sigmoid function


def relu(x):
    return np.maximum(0, x)  # implement relu function


def sigmoid_backward(curr_derive, x):
    result = sigmoid(x) * (1 - sigmoid(x))
    return curr_derive * result  # implement sigmoid derivative


def relu_backward(curr_derive, x):
    tmp_curr_derive = curr_derive.copy()
    tmp_curr_derive[x <= 0] = 0
    return tmp_curr_derive  # implement relu derivative


def make_layers(layers):
    layers_details = []
    for i in range(1, len(layers)):
        layers_details.append({"input_count": layers[i - 1],
                               "output_count": layers[i],
                               "activation": relu if i < len(layers) - 1 else sigmoid})
    return layers_details  # store layers in list and each layer detail in dictionary neuron input , output and
    # activation count


def initialize_weights(layers):
    weights_dict = {}  # store each layer weights and bias in a dictionary with "i" suffix belongs to "i"th layer
    for idx, layer in enumerate(layers):
        weights_dict[f"W{idx + 1}"] = np.random.rand(layer["output_count"], layer["input_count"]) * 0.1
        weights_dict[f"b{idx + 1}"] = np.random.rand(layer["output_count"], 1) * 0.1
    return weights_dict


def loss_function(y_predicted, y_true):
    # cross entropy loss function
    cost = -1 / data_counts * (np.dot(y_true, np.log(y_predicted).T) + np.dot(1 - y_true, np.log(1 - y_predicted).T))
    return np.squeeze(cost)


def accuracy_function(y_predicted, y_true):
    y_predicted = y_predicted.reshape(-1)  # flatten numpy array
    result = (y_predicted > 0.5) == y_true  # compare one by one each element
    return result.sum() / len(y_true)  # sum of the equal element and divide by size of data


def one_layer_forward(X_sample, weights, bias, activation):
    linear_result = np.dot(weights, X_sample) + bias  # linear multiplication -> W.X + b
    return linear_result, activation(linear_result)  # return both actual and activated neuron


def all_layer_forward(weights, layers):
    X_samples = X_train.copy()
    activated_neurons = None
    for idx, layer in enumerate(layers):
        current_layer_weights = weights[f"W{idx + 1}"]  # fetch current layer weights
        current_layer_bias = weights[f"b{idx + 1}"]  # fetch current layer bias
        current_layer_activation = layer[f"activation"]  # fetch current layer activation
        real_neurons, activated_neurons = one_layer_forward(X_samples, current_layer_weights, current_layer_bias,
                                                            current_layer_activation)  # compute forward propagation
        layer_result[f"r-n-{idx + 1}"] = real_neurons  # save result of real value neuron and layer in layer_result
        layer_result[
            f"a-n-{idx + 1}"] = activated_neurons  # save result activated value neuron and layer in layer_result
        X_samples = activated_neurons
    return activated_neurons


def predict(sample_data, weights, layers):
    X_samples = sample_data
    activated_neurons = None
    for idx, layer in enumerate(layers):
        current_layer_weights = weights[f"W{idx + 1}"]  # fetch current layer weights
        current_layer_bias = weights[f"b{idx + 1}"]  # fetch current layer bias
        current_layer_activation = layer[f"activation"]  # fetch current layer activation
        real_neurons, activated_neurons = one_layer_forward(X_samples, current_layer_weights, current_layer_bias,
                                                            current_layer_activation)
        layer_result[f"r-n-{idx + 1}"] = real_neurons  # save result of real value neuron and layer in layer_result
        layer_result[
            f"a-n-{idx + 1}"] = activated_neurons  # save result activated value neuron and layer in layer_result
        X_samples = activated_neurons
    return activated_neurons.reshape(-1)


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, current_real_neuron, previous_active_neuron,
                                      activation_backward):
    # compute one layer backward based on current derivation and neuron activated and real value
    derive_real_neuron = activation_backward(dA_curr, current_real_neuron)
    dW_curr = np.dot(derive_real_neuron, previous_active_neuron.T) / data_counts
    db_curr = np.sum(derive_real_neuron, axis=1, keepdims=True) / data_counts
    dA_prev = np.dot(W_curr.T, derive_real_neuron)
    # return the derivative so far , the gradient amount to add to W and b
    return dA_prev, dW_curr, db_curr


def all_layer_backward(y_predicted, y_true, weights, layers):
    y_true = y_true.reshape(y_predicted.shape)
    current_layer_derive = np.divide(y_true, y_predicted) - np.divide(1 - y_true, 1 - y_predicted)  # last layer derive
    current_layer_derive *= -1
    gradient_for_each_layer = {}  # store amount of gradient for each "W" and "b"
    current_layer = len(layers) - 1
    for _ in layers:
        current_layer_derive, dW_curr, db_curr = single_layer_backward_propagation(
            current_layer_derive, weights[f"W{current_layer + 1}"], weights[f"b{current_layer + 1}"],
            layer_result[f"r-n-{current_layer + 1}"], layer_result[f"a-n-{current_layer}"],
            relu_backward if current_layer < (len(layers) - 1) else sigmoid_backward)  # single backward calculate
        gradient_for_each_layer[f"dW{current_layer + 1}"] = dW_curr  # save the gradient amount in dictionary to update
        gradient_for_each_layer[f"db{current_layer + 1}"] = db_curr  # save the gradient amount in dictionary to update

        current_layer -= 1  # step one layer back

    return gradient_for_each_layer


def update_weights(weights, gradient_for_each_layer, learning_rate, layers_length):
    for i in range(1, layers_length + 1):
        weights[f"W{i}"] -= learning_rate * gradient_for_each_layer[f"dW{i}"]  # wi = wi + delta(wi)
        weights[f"b{i}"] -= learning_rate * gradient_for_each_layer[f"db{i}"]  # bi = bi + delta(bi)


data_counts, feature_counts = X.shape
layer_result = {"r-n-0": X_train, "a-n-0": X_train}  # store each layer result in dictionary


# r-n-0 : real neuron 0's layer
# a-n-0 : activated neuron 0's layer

def plot_decision_boundary(sample_X, weights, layers):
    # Set min and max values and give it some padding
    x_min, x_max = sample_X[0, :].min() - 1, sample_X[0, :].max() + 1
    y_min, y_max = sample_X[1, :].min() - 1, sample_X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, weights, layers)
    Z[Z > 0.5] = 1
    Z[Z < 0.5] = 0
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(sample_X[0, :], sample_X[1, :], c=Y, cmap=plt.cm.Spectral)
    plt.show()


def plot_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.title("loss value in each epoch")


def plot_accuracy(accuracies):
    plt.figure()
    plt.plot(accuracies)
    plt.title("accuracy value in each epoch")


def make_perceptron(epochs, learning_rate, layers):
    layers = [feature_counts, *layers, 1]  # first layers , hidden layers , output layer
    layer_details = make_layers(layers)  # detail of each layer : input , output , activation
    weights = initialize_weights(layer_details)  # give some random number to weights to initialize
    loss_outcomes = []
    accuracy_outcomes = []
    # start epoch and fit the model
    for i in range(epochs):
        # do forward propagation
        predicted_y = all_layer_forward(weights, layer_details)
        loss_outcomes.append(loss_function(predicted_y, Y))  # add loss outcome to loss list
        accuracy_outcomes.append(accuracy_function(predicted_y, Y))  # add accuracy outcome to accuracy list
        # do backward propagation
        gradient_for_each_layer = all_layer_backward(predicted_y, Y, weights, layer_details)
        update_weights(weights, gradient_for_each_layer, learning_rate, len(layer_details))  # update weights
        if i < 5:
            learning_rate /= 2  # gradually decrease learning rate until 5 epoch
    plot_accuracy(accuracy_outcomes)
    plot_loss(loss_outcomes)
    plot_decision_boundary(X_train, weights, layer_details)
    mn = 0


for i in range(3):
    print(f"for {i + 1} hidden layer")
    make_perceptron(2000, 10, [15 for _ in range(i + 1)])
    print("*********************************")

