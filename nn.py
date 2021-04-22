import numpy as np
import autograd.numpy as anp
from sklearn import datasets
from autograd import grad
from autograd.misc.flatten import flatten


# def softmax(z):
#     z -= anp.max(z, axis=0)
#     return anp.exp(z)/anp.sum(anp.exp(z), axis=0)


def softmax(z):
    z -= anp.max(z, axis=1, keepdims=True)
    return anp.exp(z)/anp.sum(anp.exp(z), axis=1, keepdims=True)


def sigmoid(z):
    return 1/(1+anp.exp(-z*1.0))


def relu(z):
    temp = np.zeros(z.shape)
    temp[z > 0] = 1
    return z*temp


def forward(W, A, X):
    out = X
    for i in range(len(W)-1):
        out = anp.concatenate([out, np.ones([len(out), 1])], axis=1)
        out = (anp.dot(W[i], out.T)).T

        if A[i] == "relu":
            out = relu(out)
        elif A[i] == "sigmoid":
            out = sigmoid(out)

    out = anp.concatenate([out, np.ones([len(out), 1])], axis=1)
    out = anp.dot(W[-1], out.T).T

    if A[-1] == "softmax":
        out = softmax(out)

    return out.T


def objective(W, A, X, y, regularization=None, L_const=1):
    y_hat = forward(W, A, X) + 1e-10

    if A[-1] == "softmax":
        z = np.zeros(y_hat.shape)
        z[y, anp.arange(y.size)] = 1

        loss = anp.sum(-z*anp.log(y_hat))

        if regularization == "l1":
            flattened, _ = flatten(W)
            sign = flattened >= 0
            loss += L_const*anp.dot(flattened, sign)
        elif regularization == "l2":
            flattened, _ = flatten(W)
            loss += L_const*anp.dot(flattened, flattened)
    else:
        loss = anp.sqrt(anp.mean(anp.square(y-y_hat)))

    return loss


class NeuralNet():
    def __init__(self, input_size, output_size, out_activation="softmax", layers=[8, ], activations=["linear", ],
                 batch_size=100, regularization=None, L_const=1):
        assert(len(layers) == len(activations))

        self.input_size = input_size
        self.output_size = output_size
        self.out_activation = out_activation
        self.batch_size = batch_size
        self.activations = activations + [out_activation, ]
        self.regularization = regularization
        self.L_const = L_const
        self.layers = layers
        self.W = self.init_weights()

    def init_weights(self):
        layers = [np.random.randn(self.layers[0], self.input_size+1), ]
        for i in range(1, len(self.layers)):
            # layers.append(np.random.randn(
            #     self.layers[i], self.layers[i-1]+1))
            layers.append(np.zeros((
                self.layers[i], self.layers[i-1]+1)))
        # layers.append(np.random.randn(
        #     self.output_size, self.layers[-1]+1))
        layers.append(np.zeros([
            self.output_size, self.layers[-1]+1]))
        return layers

    def w_grad(self, W, X, y):
        return objective(W, self.activations, X, y, self.regularization, self.L_const)

    def predict(self, X):
        y = forward(self.W, self.activations, X)
        if self.out_activation == "softmax":
            return np.argmax(y, axis=0)
        return y

    def fit(self, X, y, iterations=1e3, lr=1e-2, verbose=True, log_interval=1e1):

        for i in range(int(iterations+1)):
            # if i > 4e2:
            #     lr = 1e-2

            if (i) % int(log_interval) == 0 and verbose:
                loss = objective(self.W, self.activations, X, y,
                                 regularization=self.regularization)
                print(f"For Iteration:{i:3d} | loss: {loss:3f}")

            for j in range(0, len(X), self.batch_size):
                X_batch = X[j:j+self.batch_size]
                y_batch = y[j:j+self.batch_size]

                dW = grad(self.w_grad)(
                    self.W, X_batch.astype(float), y_batch)

                for i in range(len(self.W)):
                    self.W[i] -= lr/self.batch_size*dW[i]


if __name__ == '__main__':
    '''
    digits = datasets.load_digits()
    X = digits.data.copy()
    # X /= np.max(X)
    y = digits.target
    print(
        f"The dataset contains {len(X)} samples of size:{X[0].shape[0]} and have {len(np.unique(y))} unique classes")

    nn = NeuralNet(len(X[0]), len(np.unique(y)))
    # forward(nn.W, nn.activations, X)
    objective(nn.W, nn.activations, X, y)
    y_hat = nn.predict(X)
    accuracy = np.sum(y_hat == y)/len(y)*100
    nn.fit(X, y)
    print(f"Accuracy before training: {accuracy:.3f}")
    y_hat = nn.predict(X)
    accuracy = np.sum(y_hat == y)/len(y)*100
    print(f"Accuracy after training: {accuracy:.3f}")
    print("hello")
    '''
    boston = datasets.load_boston()
    X = boston.data.copy()
    y = boston.target.copy()

    nn = NeuralNet(len(X[0]), 1, out_activation="linear",
                   layers=[10, ], activations=["relu", ])
    nn.fit(X, y, lr=1e-4, iterations=5e2)
    print("hello")
