

from Binnary_LR import LR, objective
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from metrics import accuracy

data = pd.read_csv("./Breast_Cancer_Dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

sc = MinMaxScaler()
X = pd.DataFrame(sc.fit_transform(X))


REG = 'l2'


def kfold(X, y, folds=3, L_const=np.linspace(0, 3, 6), verbose=False):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    models = []
    accuracies = []
    chunk = int(len(X)//folds)

    for constant in L_const:
        print(f"\nChecking for lambda: {constant}")
        regressor = LR(X.shape[-1], L_const=constant,
                       regularization=REG, autograd=True)

        for fold in range(folds):
            if verbose:
                print(f"For fold {fold+1}:")

            # Trains a model for each fold
            indices = range(fold*chunk, (fold+1)*chunk)
            curr_fold = pd.Series([False for i in range(len(X))])
            curr_fold.loc[indices] = True

            X_train, y_train = X[~curr_fold].reset_index(
                drop=True).values, y[~curr_fold].reset_index(drop=True).values
            X_test, y_test = X[curr_fold].reset_index(
                drop=True).values, y[curr_fold].reset_index(drop=True).values

            regressor.fit(X_train, y_train, iterations=50e2,
                          log_interval=1e3, verbose=verbose)
            y_hat = regressor.predict(X_test)

            # print(confusion_matrix(y_test, y_hat))
        y_hat = regressor.predict(X)
        y_hat = y_hat >= 0.5
        a = accuracy(y_hat, y)
        print(f"Accuracy for lambda: {constant} is {a:.3f}")
        accuracies.append(a)
        models.append(regressor)

    print(f"Best Accuracy: {np.max(accuracies)}")
    index = np.argmax(accuracies)
    return models[index]


kfold(X, y)
print("hello")
'''


before = objective(regressor.W, regressor.b, X, y)/len(X)
print(
    f"Average loss before training: {before:3f}")
print(
    f"Average loss after training: {objective(regressor.W,regressor.b,X,y)/len(X):3f}")

print("Confusion Matrix:")

y_hat = regressor.predict(X)
y_hat = y_hat >= 0.5

print(confusion_matrix(y, y_hat))
print("Accuracy: ", accuracy(y_hat, y))
'''
