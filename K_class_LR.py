import numpy as np
import autograd.numpy as anp
from sklearn import datasets
from autograd import grad
from autograd.misc.flatten import flatten


def softmax(z):
    z -= anp.max(z, axis=0)
    return anp.exp(z)/anp.sum(anp.exp(z), axis=0)


def forward(W, b, X):
    return softmax(anp.dot(W, X.T) + b)


def select(y_hat, y):
    return (y_hat[y])


def backward(W, b, X, y):
    y_hat = forward(W, b, X)
    z = np.zeros(y_hat.shape)
    z[y, np.arange(y.size)] = 1
    loss = -(z-y_hat)
    return loss@X, np.sum(loss, axis=1, keepdims=True)


def objective(W, b, X, y, regularization=None, L_const=1):
    y_hat = forward(W, b, X)
    z = np.zeros(y_hat.shape)
    z[y, np.arange(y.size)] = 1

    loss = anp.sum(-z*anp.log(y_hat))

    if regularization == "l1":
        flattened, _ = flatten(W)
        sign = flattened >= 0
        loss += L_const*anp.dot(flattened, sign)
    elif regularization == "l2":
        flattened, _ = flatten(W)
        loss += L_const*anp.dot(flattened, flattened)

    return loss


class K_class_LogisticRegressor():
    def __init__(self, num_features, num_classes,
                 bias=True,
                 regularization=None,
                 autograd=False,
                 batch_size=100,
                 L_const=1):

        self.W = np.zeros((num_classes, num_features))
        self.b = np.zeros((num_classes, 1)) if bias else 0
        self.num_classes = num_classes
        self.num_features = num_features
        self.regularization = regularization
        self.autograd = autograd
        self.batch_size = batch_size
        self.L_const = L_const

    def w_grad(self, W, X, y):
        return objective(W, self.b, X, y, self.regularization, self.L_const)

    def b_grad(self, b, X, y):
        return objective(self.W, b, X, y, self.regularization, self.L_const)

    def predict(self, X):
        y = forward(self.W, self.b, X)
        return np.argmax(y, axis=0)

    def fit(self, X, y, iterations=1e2, lr=1e-2, verbose=True, log_interval=1e1):

        for i in range(int(iterations+1)):

            if (i) % int(log_interval) == 0 and verbose:
                loss = objective(self.W, self.b, X, y,
                                 regularization=self.regularization)
                print(f"For Iteration:{i:3d} | loss: {loss:3f}")

            for j in range(0, len(X), self.batch_size):
                X_batch = X[j:j+self.batch_size]
                y_batch = y[j:j+self.batch_size]

                if self.autograd:
                    dW = grad(self.w_grad)(self.W, X_batch, y_batch)
                    db = grad(self.b_grad)(self.b, X_batch, y_batch)
                else:
                    dW, db = backward(self.W, self.b, X_batch, y_batch)

                self.W -= lr/self.batch_size*dW
                self.b -= lr/self.batch_size*db


if __name__ == '__main__':
    digits = datasets.load_digits()
    X = digits.data.copy()
    # X /= np.max(X)
    y = digits.target
    print(
        f"The dataset contains {len(X)} samples of size:{X[0].shape[0]} and have {len(np.unique(y))} unique classes")

    K_lr = K_class_LogisticRegressor(len(X[0]), len(np.unique(y)))
    y_hat = K_lr.predict(X)
    accuracy = np.sum(y_hat == y)/len(y)*100
    K_lr.fit(X, y)
    print(f"Accuracy before training: {accuracy:.3f}")
    y_hat = K_lr.predict(X)
    accuracy = np.sum(y_hat == y)/len(y)*100
    print(f"Accuracy after training: {accuracy:.3f}")
    print("hello")
