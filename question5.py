

from nn import NeuralNet, objective
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
from metrics import accuracy
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

digits = datasets.load_digits()
X = digits.data.copy()
y = digits.target


model = NeuralNet(len(X[0]), len(np.unique(y)))


def kfold(X, y, folds=4):
    kf = KFold(n_splits=folds, shuffle=True)

    assert(len(X) == len(y))
    assert(len(X) > 0)

    chunk = int(len(X)//folds)

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"For fold {fold+1}:")
        # Trains a model for each fold

        X_train, y_train = X[train_index], y[train_index]
        model.fit(X_train, y_train, iterations=1e2, log_interval=1e1)
        print()


# Q5 Part A
y_hat = model.predict(X)
accuracy = np.sum(y_hat == y)/len(y)*100
kfold(X, y)

print(f"Accuracy before training: {accuracy:.3f}")
y_hat = model.predict(X)
accuracy = np.sum(y_hat == y)/len(y)*100
print(f"Accuracy after training: {accuracy:.3f}")


# plt.ion()
df_cm = pd.DataFrame(confusion_matrix(y, y_hat),
                     index=range(10), columns=range(10))
sns.heatmap(df_cm, annot=True)
plt.title('Digits Confusion Matrix')
plt.show()

# # Q2 Part D
# pca = PCA(n_components=2, svd_solver='arpack')
# X_pca = pca.fit_transform(X)
# plt.figure()

# for i in np.unique(y):
#     plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f"{i}")
# plt.legend()
# plt.title("Digits dataset visualised using PCA")
# plt.show(block=True)
