from Layer import Layer
from functions import HBTangents, Sigmoid, ReLU, Constant, MSE


class Model:
    def __init__(self):
        self.layers = []
        # Можно использовать Sigmoid вместо HBTangents, чтобы результат около 0 был правильным,
        # но это ухудшит результат
        self.layers.append(Layer(1, 100, HBTangents))
        self.layers.append(Layer(100, 200, ReLU))
        self.layers.append(Layer(200, 100, HBTangents))
        self.layers.append(Layer(100, 1, Constant))

        self.costFunction = MSE

    def Train(self, dataset, target, learning_rate, epochNum):
        epochsErrors = []
        for epoch in range(epochNum):
            error = 0
            for sample, targetValue in zip(dataset, target):
                inputs = sample
                for layer in self.layers:
                    inputs = layer.ForwardPropagation(inputs)
                error += self.costFunction(targetValue, inputs)
                backwardError = self.costFunction(targetValue, inputs, True)
                for layer in reversed(self.layers):
                    backwardError = layer.BackwardPropagation(backwardError, learning_rate)
            epochsErrors.append(error / len(dataset))
            if epoch % 10 == 9:
                print(f"Epoch #{epoch + 1}. Error: {round(epochsErrors[epoch], 6)}")
        return epochsErrors

    def PredictValues(self, inputValue):
        for layer in self.layers:
            inputValue = layer.ForwardPropagation(inputValue)
        return [inputValue]
