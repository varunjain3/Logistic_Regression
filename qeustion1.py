

from tools import LogisticRegression, accuracy, precision, recall, objective
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./Breast_Cancer_Dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

sc = MinMaxScaler()
X = pd.DataFrame(sc.fit_transform(X))


regressor = LogisticRegression(X.shape[-1])


def kfold(X, y, folds=5):
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
        X_test, y_test = X[curr_fold].reset_index(
            drop=True).values, y[curr_fold].reset_index(drop=True).values

        regressor.fit(X_train, y_train, iterations=4e2, log_interval=1e2)
        y_hat = regressor.predict(X_test)

        # print('Accuracy: ', accuracy(y_hat, y_test))
        # for cls in y.unique():
        #     print('Precision: ', precision(y_hat, y_test, cls))
        #     print('Recall: ', recall(y_hat, y_test, cls))
        print()


before = objective(regressor.W, regressor.b, X, y)/len(X)
kfold(X, y)
print(
    f"Average loss before training: {before:3f}")

print(
    f"Average loss after training: {objective(regressor.W,regressor.b,X,y)/len(X):3f}")
