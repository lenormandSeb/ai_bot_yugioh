import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score

def initialisation(x):
    w = np.random.randn(x.shape[1], 1)
    b = np.random.randn(1)
    return (w, b)

def prediction(x, w, b):
    A = model(x, w, b)
    return A >= 0.5

def model(x, w, b):
    Z = x.dot(w) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss_neuron(A, y):
    exponentiel = 1e-15
    # exponentiel = 0
    return 1 / len(y) * np.sum(-y * np.log(A + exponentiel) - (1 - y) * np.log(1 - A + exponentiel))

def gradient_neuron(A, x, y):
    dw = 1 / len(y) * np.dot(x.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dw, db)

def update_neuron(dw, db, w, b, learning):
    w = w - learning * dw
    b = b - learning * db
    return (w, b)

def simple_neuron(x, y, learning = 0.1, n_iter = 100):
    w, b = initialisation(x)
    loss = []
    for i in range(n_iter):
        A = model(x, w, b)
        loss.append(log_loss_neuron(A, y))
        dw, db = gradient_neuron(A, x, y)
        w, b = update_neuron(dw, db, w, b, learning)
    
    plt.plot(loss)
    plt.show()

    return (w, b)