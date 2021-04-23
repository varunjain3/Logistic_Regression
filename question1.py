

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import accuracy
from Binnary_LR import LR, objective
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# Data Preprocessing
data = pd.read_csv("./Breast_Cancer_Dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
sc = MinMaxScaler()
X = pd.DataFrame(sc.fit_transform(X))


regressor = LR(X.shape[-1], regularization="l1")


def kfold(X, y, folds=3):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    accuracies = {}
    chunk = int(len(X)//folds)

    for fold in range(folds):
        print(f"For fold {fold+1}:")
        # Trains a model for each fold
        indices = range(fold*chunk, (fold+1)*chunk)
        curr_fold = pd.Series([False for i in range(len(X))])
        curr_fold.loc[indices] = True

        X_train, y_train = X[~curr_fold].reset_index(
            drop=True).values, y[~curr_fold].reset_index(drop=True).values

        regressor.fit(X_train, y_train, iterations=5e3, log_interval=2e3)
        print()


def plot_db(X, y, f1, f2):
    # To plot for decision Boundaries
    X = np.array(X)[:, [f1, f2]]
    y = np.array(y)

    regressor = LR(X.shape[-1])
    regressor.fit(X, y, iterations=1e4, lr=1e-1, log_interval=3e3)

    fig, ax = plt.subplots(figsize=(9, 7))

    color = ["r", "b", "g"]
    for i in np.unique(y):
        ax.scatter(X[y == i, 0], X[y == i, 1],
                   c=color[i], label=f"Class:{i}")
    ax.set_ylabel("y")
    ax.set_xlabel("X")
    ax.legend()

    x1_min, x1_max = np.min(X[:, 0]) - 0.25, np.max(X[:, 0]) + 0.25
    x2_min, x2_max = np.min(X[:, 1]) - 0.25, np.max(X[:, 1]) + 0.25
    d = 0.01
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, d),
                           np.arange(x2_min, x2_max, d))
    y = regressor.predict(np.c_[xx1.ravel(), xx2.ravel()])
    y = y.reshape(xx1.shape)
    surf = ax.contourf(
        xx1, xx2, y, cmap=plt.cm.RdYlBu, alpha=.2)
    ax.set_ylabel(f"feauter_{f2}")
    ax.set_xlabel(f"Feature_{f1}")
    ax.set_title(f"Decision Boundary for feature {f1} and {f2}", fontsize=16)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    fig.savefig("./figures/Q1_DecisionBoundary.png")

    return fig


# For K-fold creoss Validation
before = objective(regressor.W, regressor.b, X, y)/len(X)
kfold(X, y)
print(f"Average loss before training: {before:3f}")
print(
    f"Average loss after training: {objective(regressor.W,regressor.b,X,y)/len(X):3f}")

print("Confusion Matrix:")
y_hat = regressor.predict(X)
print(confusion_matrix(y, y_hat))
print("Accuracy: ", accuracy(y_hat, y))

# Plotting decision boundaryhaa
fig = plot_db(X, y, 1, 2)
