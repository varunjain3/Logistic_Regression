
import numpy as np
import pandas as pd
import autograd.numpy as anp

from autograd import grad
from metrics import accuracy
from autograd.misc.flatten import flatten
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix


def sigmoid(z):
    return 1/(1 + anp.exp(-z*1.0))


def forward(W, b, X, a_type="sigmoid"):
    if a_type == "sigmoid":
        return sigmoid(anp.dot(W, X.T) + b)
    else:
        raise Exception("Not Implemented")


# Function to caculate backpropagation for Logistic Regression
def backward(W, b, X, y):
    y_hat = forward(W, b, X)
    dJ_theta = (y_hat-y)@X
    dJ_bias = np.mean((y_hat-y))
    return dJ_theta, dJ_bias


# Objective Loss function with L1 and L2 regularization
def objective(W, b, X, y, regularization="unregularized", L_const=1):
    y_hat = forward(W, b, X)
    loss = -1*anp.sum(y*anp.log(y_hat) + (1-y)*anp.log(1-y_hat))

    if regularization == "l1":
        flattened, _ = flatten(W)
        sign = flattened >= 0
        loss += L_const*anp.dot(flattened, sign)
    elif regularization == "l2":
        flattened, _ = flatten(W)
        loss += L_const*anp.dot(flattened, flattened)

    return loss


class LR():
    # Main Logistic Regression Class
    def __init__(self, num_features, bias=True,
                 activation='sigmoid',
                 regularization="unregularized",
                 autograd=False,
                 L_const=1):
        self.W = np.zeros(num_features)
        self.b = np.zeros(1) if bias else 0
        self.a_func = activation
        self.regularization = regularization
        self.X = None
        self.y = None
        self.L_const = L_const
        self.autograd = autograd

    def w_grad(self, W):
        return objective(W, self.b, self.X, self.y, self.regularization, self.L_const)

    def b_grad(self, b):
        return objective(self.W, b, self.X, self.y, self.regularization, self.L_const)

    # To fit the classifier
    def fit(self, X, y, iterations=1e2, lr=1e-2, verbose=True, log_interval=2):
        self.X = X
        self.y = y

        for i in range(int(iterations)):

            if (i) % int(log_interval) == 0 and verbose:
                loss = objective(self.W, self.b, self.X, self.y,
                                 regularization=self.regularization)
                print(f"For Iteration:{i:3d} | loss: {loss:3f}")

            if self.autograd:
                dW = grad(self.w_grad)(self.W)
                db = grad(self.b_grad)(self.b)
            else:
                dW, db = backward(self.W, self.b, self.X, self.y)
            self.W -= lr*dW
            self.b -= lr*db

    # To predict the values
    def predict(self, X):
        out = forward(self.W, self.b, X)
        out = out >= 0.5
        return out


if __name__ == '__main__':
    N_ITER = 1e5
    data = pd.read_csv("./Breast_Cancer_Dataset.csv")
    X = normalize(data.iloc[:, :-1].values)
    y = data.iloc[:, -1].values

    # Defining the Classfier
    lr = LR(P)
    lr.fit(X, y, iterations=N_ITER, log_interval=int(N_ITER//5), lr=0.03)

    y_hat = lr.predict(X)

    print(confusion_matrix(y, y_hat))
    print("Accuracy: ", accuracy(y_hat, y))
