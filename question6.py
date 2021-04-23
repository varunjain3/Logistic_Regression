

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nn import NeuralNet
from sklearn import datasets
from metrics import accuracy, rmse
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def kfold(X, y, folds=3):
    kf = KFold(n_splits=folds, shuffle=True)

    assert(len(X) == len(y))
    assert(len(X) > 0)

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"For fold {fold+1}:")
        # Trains a model for each fold

        X_train, y_train = X[train_index], y[train_index]
        model.fit(X_train, y_train,
                  lr=params["lr"], iterations=params["iterations"], log_interval=params["log_interval"])
        print()


if __name__ == '__main__':
    # Q5 Part A Digits
    digits = datasets.load_digits()
    X = digits.data.copy()
    y = digits.target
    model = NeuralNet(len(X[0]), len(np.unique(y)))
    params = {"lr": 1e-2, "iterations": 1e2, "log_interval": 1e1}
    y_hat = model.predict(X)
    accuracy = np.sum(y_hat == y)/len(y)*100
    kfold(X, y)

    print(f"Accuracy before training: {accuracy:.3f}")
    y_hat = model.predict(X)
    accuracy = np.sum(y_hat == y)/len(y)*100
    print(f"Accuracy after training: {accuracy:.3f}")

    plt.ion()
    df_cm = pd.DataFrame(confusion_matrix(y, y_hat),
                         index=range(10), columns=range(10))
    sns.heatmap(df_cm, annot=True)
    plt.title('Digits Confusion Matrix')

    # Q5 Part B Boston Housing
    boston = datasets.load_boston()
    X = boston.data.copy()
    y = boston.target.copy()

    model = NeuralNet(len(X[0]), 1, out_activation="linear",
                      layers=[10, ], activations=["relu", ])
    params = {"lr": 1e-4, "iterations": 5e2, "log_interval": 1e2}
    y_hat = model.predict(X)
    kfold(X, y)
    print(f"Rmse before Training {rmse(y_hat,y)}")
    y_hat = model.predict(X)
    print(f"Rmse After Training {rmse(y_hat,y)}")
    plt.show(block=True)
