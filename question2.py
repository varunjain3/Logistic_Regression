

from matplotlib import pyplot as plt
from Binnary_LR import LR, objective
import numpy as np
import pandas as pd

from metrics import accuracy
from sklearn.preprocessing import MinMaxScaler

# Preprocessing
data = pd.read_csv("./Breast_Cancer_Dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
sc = MinMaxScaler()
X = pd.DataFrame(sc.fit_transform(X))


REG = 'l2'


def kfold(X, y, folds=3, L_const=np.linspace(0.1, 1.5, 10), verbose=False):
    assert(len(X) == len(y))
    assert(len(X) > 0)

    models = []
    accuracies = []
    chunk = int(len(X)//folds)

    # For different values of L
    for constant in L_const:
        print(f"\nChecking for lambda: {constant}")
        regressor = LR(X.shape[-1], L_const=constant,
                       regularization=REG, autograd=True)

        # For n foldds
        for fold in range(folds):
            if verbose:
                print(f"For fold {fold+1}:")

            # Trains a model for each fold
            indices = range(fold*chunk, (fold+1)*chunk)
            curr_fold = pd.Series([False for i in range(len(X))])
            curr_fold.loc[indices] = True

            X_train, y_train = X[~curr_fold].reset_index(
                drop=True).values, y[~curr_fold].reset_index(drop=True).values

            regressor.fit(X_train, y_train, iterations=5e2,
                          log_interval=1e3, verbose=verbose)

        y_hat = regressor.predict(X)
        a = accuracy(y_hat, y)
        print(f"Accuracy for lambda: {constant} is {a:.3f}")
        accuracies.append(a)
        models.append(regressor)
    index = np.argmax(accuracies)
    print(
        f"Best Accuracy: {np.max(accuracies)} for Lambda:{L_const[index]}")

    plt.figure(figsize=(10, 5))
    plt.plot(L_const, accuracies)
    plt.title("Accuracy vs Lambda")
    plt.xlabel("Values of Lambda")
    plt.ylabel("Accuracy")
    plt.savefig("./figures/Q2_Acc_v_lambda.png")
    return models[index]


model = kfold(X, y)

# For Feature Importance
plt.figure(figsize=(8, 6))
plt.bar(data.columns.values[:len(model.W)], np.abs(model.W))
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.savefig("./figures/Q2_FeatureImp.png")
