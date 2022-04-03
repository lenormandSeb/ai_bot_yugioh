import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score

def init(n0, n1, n2):
    w1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)

    w2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    w3 = np.random.randn(n1, n1)
    b3 = np.random.randn(n1, 1)

    w4 = np.random.randn(n2, n1)
    b4 = np.random.randn(n2, 1)

    parameter = {
        'w1' : w1,
        'w2' : w2,
        'w3' : w3,
        'w4' : w4,
        'b1' : b1,
        'b2' : b2,
        'b3' : b3,
        'b4' : b4,
    }

    return parameter

def forward_propagation(x, parameter):
    w1 = parameter['w1']
    w2 = parameter['w2']
    w3 = parameter['w3']
    w4 = parameter['w4']
    b1 = parameter['b1']
    b2 = parameter['b2']
    b3 = parameter['b3']
    b4 = parameter['b4']

    Z1 = w1.dot(x) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    Z2 = w2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    Z3 = w3.dot(A1) + b3
    A3 = 1 / (1 + np.exp(-Z3))

    Z4 = w4.dot(A3) + b4
    A4 = 1 / (1 + np.exp(-Z4))

    activations = {
        'A1' : A1,
        'A2' : A2,
        'A3' : A3,
        'A4' : A4,
    }

    return activations

def log_loss(A, y):
    expo = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + expo) - (1 - y) * np.log(1 - A + expo))

def back_propagation(x, y, activations, parameter):

    A1 = activations['A1']
    A2 = activations['A2']
    A3 = activations['A3']
    A4 = activations['A4']
    w2 = parameter['w2']
    w3 = parameter['w3']
    w4 = parameter['w4']

    m = y.shape[1]

    dz4 = A4 - y
    dw4 = 1 / m * dz4.dot(A3.T)
    db4 = 1 / m * np.sum(dz4, axis=1, keepdims=True)

    dz3 = np.dot(w4.T, dz4) * A3 * (1 - A3)
    dw3 = 1 / m * dz3.dot(A2.T)
    db3 = 1 / m * np.sum(dz3, axis=1, keepdims=True)

    dz2 = np.dot(w3.T, dz3) * A2 * (1 - A2)
    dw2 = 1 / m * dz2.dot(A2.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2, dz2) * A1 * (1 - A1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)


    gradient = {
        'dw1' : dw1,
        'db1' : db1,
        'dw2' : dw2,
        'db2' : db2,
        'dw3' : dw3,
        'db3' : db3,
        'dw4' : dw4,
        'db4' : db4,
    }

    return gradient

def update(gradient, parameter, learning):
    dw1 = gradient["dw1"]
    dw2 = gradient["dw2"]
    dw3 = gradient["dw3"]
    dw4 = gradient["dw4"]

    db1 = gradient["db1"]
    db2 = gradient["db2"]
    db3 = gradient["db3"]
    db4 = gradient["db4"]

    w1 = parameter["w1"]
    b1 = parameter["b1"]
    w2 = parameter["w2"]
    b2 = parameter["b2"]
    w3 = parameter["w3"]
    b3 = parameter["b3"]
    w4 = parameter["w4"]
    b4 = parameter["b4"]

    w1 = w1 - learning * dw1
    b1 = b1 - learning * db1
    w2 = w2 - learning * dw2
    b2 = b2 - learning * db2
    w3 = w3 - learning * dw3
    b3 = b3 - learning * db3
    w4 = w4 - learning * dw4
    b4 = b4 - learning * db4
    
    parameter = {
        'w1' : w1,
        'w2' : w2,
        'w3' : w3,
        'w4' : w4,
        'b1' : b1,
        'b2' : b2,
        'b3' : b3,
        'b4' : b4,
    }

    return parameter

def predict(x, parameter):
    activations = forward_propagation(x, parameter)
    A4 = activations['A4']
    return A4 >= 0.5

def neuron(x, y, n1, learning = 0.01, n_iter = 8000):
    n0 = x.shape[0]
    n2 = y.shape[0]
    parameter = init(n0, n1, n2)

    lost = []
    ac = []
    for i in range(n_iter):
        A = forward_propagation(x,parameter)
        gradient = back_propagation(x, y, A, parameter)
        parameter = update(gradient, parameter, learning)
        if i %10 == 0:
            lost.append(log_loss(y, A["A4"]))
            y_pred = predict(x, parameter)
            current_accu = accuracy_score(y.flatten(), y_pred.flatten())
            ac.append(current_accu)

    print(current_accu)
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.plot(lost, label="lost")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ac, label="ac")
    plt.legend()

    plt.show()
    return parameter
