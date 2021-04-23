

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from metrics import accuracy
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from K_class_LR import K_class_LogisticRegressor
from sklearn.model_selection import StratifiedKFold

# Preprocessing
digits = datasets.load_digits()
X = digits.data.copy()
y = digits.target


# Defining Classifier
K_lr = K_class_LogisticRegressor(len(X[0]), len(np.unique(y)))


def kfold(X, y, folds=4):
    kf = StratifiedKFold(n_splits=folds, shuffle=True)

    assert(len(X) == len(y))
    assert(len(X) > 0)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"For fold {fold+1}:")

        # Trains a model for each fold
        X_train, y_train = X[train_index], y[train_index]
        K_lr.fit(X_train, y_train, iterations=2e0, log_interval=2e0)
        print()


# Q2 Part B
y_hat = K_lr.predict(X)
accuracy = np.sum(y_hat == y)/len(y)*100
kfold(X, y)

print(f"Accuracy before training: {accuracy:.3f}")
y_hat = K_lr.predict(X)
accuracy = np.sum(y_hat == y)/len(y)*100
print(f"Accuracy after training: {accuracy:.3f}")

plt.ion()
df_cm = pd.DataFrame(confusion_matrix(y, y_hat),
                     index=range(10), columns=range(10))
sns.heatmap(df_cm, annot=True)
plt.title('Digits Confusion Matrix')
plt.savefig("./figures/Q3_DigitsCM.png")

# Q2 Part D
pca = PCA(n_components=2, svd_solver='arpack')
X_pca = pca.fit_transform(X)
plt.figure()

for i in np.unique(y):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=f"{i}")
plt.legend()
plt.title("Digits dataset visualised using PCA")
plt.savefig("./figures/Q3_DigitsPCA.png")
plt.show(block=True)
