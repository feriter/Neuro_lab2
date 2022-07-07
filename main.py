import math
import numpy as np
from matplotlib import pyplot as plt

import Model as m


def f(x):  # function to approximate
    return math.cos(x)


def activate(x, deriv=False):
    if deriv:
        return activate(x) * (1 - activate(x))
    return 1 / (1 + np.exp(-x))


def generate_dataset(amount):
    train_x = []
    train_y = []
    for _ in range(amount):
        x = 8 * math.pi * np.random.rand() - 4 * math.pi  # [-4pi;4pi) values
        y = f(x)
        train_x.append(x)
        train_y.append(y)
    return np.array(train_x), np.array(train_y)


def DrawPlots(epochNum, epochErrors, dataset, target, predictions):
    fig = plt.figure()

    plt.scatter([i for i in range(epochNum)], epochErrors, s=10, color='#ff7373')
    plt.legend(["error"], bbox_to_anchor=(0, 1, 1, 0), loc="lower center", ncol=2)
    plt.show()
    fig.savefig('plots/error.png', dpi=150)

    fig = plt.figure()
    plt.scatter(dataset, target, s=10, color='#8d4dca')
    plt.scatter(dataset, predictions, s=10, color='#ff9c00')
    plt.legend(["target", "predicted"], bbox_to_anchor=(0, 1, 1, 0), loc="lower center", ncol=2)
    plt.show()
    fig.savefig('plots/predictions.png', dpi=150)


if __name__ == "__main__":
    learning_rate = 0.0001
    epochs = 150
    learning_part = 0.7
    test_amount = 1000
    learning_amount = int(learning_part * test_amount)

    # Датасет
    X, Y = generate_dataset(test_amount)

    # Обучающая выборка 70%
    learn_X = X[:learning_amount]
    learn_Y = Y[:learning_amount]

    model = m.Model()

    # Обучение модели
    errors = model.Train(learn_X, learn_Y, learning_rate, epochs)

    # Валидационная выборка 30%
    validate_X = X[learning_amount:]
    validate_Y = Y[learning_amount:]

    # Числитель и знаменатель для подсчета R^2
    numer = 0
    denom = 0
    y_ = np.mean(validate_Y)
    predictions = []
    for i in zip(validate_X, validate_Y):
        xi = i[0]
        yi = i[1]
        axi = model.PredictValues(xi)
        numer += (axi - yi) * (axi - yi)
        denom += (yi - y_) * (yi - y_)
        predictions.append(np.asarray(axi).item())
    # Графики ошибки и сравнения истинных значений с полученными от модели
    DrawPlots(epochs, errors, validate_X, validate_Y, predictions)
    R_2 = 1 - numer / denom
    print('R^2: ' + str(R_2))
