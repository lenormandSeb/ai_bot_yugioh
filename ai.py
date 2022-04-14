import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score

def init(dimensions):
    parameter = {}

    for c in range(1, len(dimensions)):
        parameter['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parameter['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parameter

def forward_propagation(x, parameter):
    activations = {'A0' : x}

    C = len(parameter) // 2

    for c in range(1, C + 1):
        Z = parameter['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameter['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations

def log_loss(A, y):
    expo = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + expo) - (1 - y) * np.log(1 - A + expo))

def back_propagation(y, activations, parameter):
    gradient = {}
    m = y.shape[1]
    C = len(parameter) // 2

    dz = activations['A' + str(C)] - y

    for c in reversed(range(1, C + 1)):
        gradient['dw' + str(c)] = 1 / m * np.dot(dz, activations['A' + str(c - 1)].T)
        gradient['db' + str(c)] = 1 / m * np.sum(dz, axis=1, keepdims=True)
        if c > 1:
            dz = np.dot(parameter['W' + str(c)].T, dz) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

    return gradient

def update(gradient, parameter, learning):
    C = len(parameter) // 2

    for c in range(1, C + 1):
        parameter['W' + str(c)] = parameter['W' + str(c)] - learning - gradient['dw' + str(c)]
        parameter['b' + str(c)] = parameter['b' + str(c)] - learning - gradient['db' + str(c)]

    return parameter

def predict(x, parameter):
    activations = forward_propagation(x, parameter)
    C = len(parameter) // 2
    Af = activations['A' + str(C)]
    return Af >= 0.5

def neuron_network(x, y, hidden_layer = (32,32,32), learning = 0.01, n_iter = 8000):
    dimensions = list(hidden_layer)
    dimensions.insert(0, x.shape[0])
    dimensions.append(y.shape[0])

    parameter = init(dimensions)

    train_loss = []
    train_cout = []

    for i in range(n_iter):
        activations = forward_propagation(x, parameter)
        gradient = back_propagation(y, activations, parameter)
        parameter = update(gradient, parameter, learning)

        if i %10 == 0:
            C = len(parameter) // 2
            train_loss.append(log_loss(y, activations['A' + str(C)]))
            y_pred = predict(x, parameter)
            current = accuracy_score(y.flatten(), y_pred.flatten())
            train_cout.append(current)

    fix, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,4))
    ax[0].plot(train_loss, label='train loss')
    ax[0].legend()

    ax[1].plot(train_cout, label='train cout')
    ax[1].legend()
    plt.show()
    return parameter
