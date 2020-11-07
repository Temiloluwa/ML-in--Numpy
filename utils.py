import numpy as np

def tanh(x):
    return np.tanh(x)


def softmax(x):
    x = np.exp(x)
    return x/np.sum(x, axis=0, keepdims=True)


def sigmoid(x):
    return 1/(1 + np.exp(-x))