import numpy as np


class Layer:
    def __init__(self, previousDim, currentDim, activationFunction):
        self.inputs = None
        self.linearCombination = None
        self.weights = np.random.uniform(-1, 1, (previousDim, currentDim))
        self.activationFunction = activationFunction

    def ForwardPropagation(self, inputs):
        self.inputs = inputs
        self.linearCombination = np.dot(self.inputs, self.weights)
        return self.activationFunction(self.linearCombination)

    def BackwardPropagation(self, backwardError, learning_rate):
        backwardError = self.activationFunction(self.linearCombination, True) * backwardError
        self.weights -= learning_rate * np.dot(self.inputs.T, backwardError)
        delta = np.dot(backwardError, self.weights.T)
        return delta
