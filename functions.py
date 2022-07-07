import numpy as np


def MSE(predicted, target, deriv=False):
    if deriv:
        return 2 * (target - predicted) / predicted.size
    return np.sum(np.power((target - predicted), 2))


def Sigmoid(x, deriv=False):
    if deriv:
        Sigmoid(x) * (1 - Sigmoid(x))
    return 1 / (1 + np.exp(-x))


def HBTangents(x, deriv=False):
    if deriv:
        return 1 - np.power(HBTangents(x), 2)
    return 2 * Sigmoid(2 * x) - 1


def ReLU(x, deriv=False):
    if deriv:
        return np.heaviside(x, 0)
    return np.maximum(0, x)


def Constant(x, deriv=False):
    if deriv:
        return 1
    return x
