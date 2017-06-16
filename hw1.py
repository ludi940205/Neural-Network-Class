from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt


def load_images():
    # load the MNIST data set
    nTrain = 20000
    nTest = 2000
    mndata = MNIST('.')
    mndata.load_training()
    mndata.load_testing()

    # using numpy array form to store the train and test sets
    # and transform the data type to double
    train_images = np.array(mndata.train_images[:nTrain]).T / 255.
    train_labels = np.array(mndata.train_labels[:nTrain]).T
    test_images = np.array(mndata.test_images[:nTest]).T / 255.
    test_labels = np.array(mndata.test_labels[:nTest]).T
    return train_images, train_labels, test_images, test_labels


# pre-process the data so that each pixel will have roughly zero mean and unit variance.
def whitening(images):
    m = np.mean(images, axis=1)
    s = np.std(images, axis=1)
    images = images.T - m
    images /= (s + 0.1)
    images = images.T
    return images


# randomly shuffle the data
def shuffle(X, t):
    idx = np.random.permutation(t.size)
    X, t = X[:, idx], t[idx]
    return X, t


# insert an extra dimension for the bias term
def insert_bias(X):
    X = np.insert(X, X.shape[0], 1, axis=0)
    return X


# gradient descent
def gradient_descent(method, W, learning_rate, anneal, early_stop, mini_batch):
    d, n = method.X.shape
    batch_size = n / mini_batch
    weights = []
    for i in xrange(200):
        start, end = 0, batch_size
        for j in xrange(mini_batch):
            X_batch, t_batch = method.X[:, start:end], method.t[start:end]

            train_loss, train_error = method.eval_loss_and_error(method.X, method.t, W)
            test_loss, test_error = method.test(test_images, test_labels, W)

            dW = method.eval_gradient(X_batch, t_batch, W)

            learning_rate = learning_rate if not anneal else \
                learning_rate / (1. + i / anneal) if i * mini_batch + j > 5 else 1e-1

            weights.append(np.array(W))

            if early_stop:
                holdout_loss, holdout_error = method.eval_loss_and_error(method.X_holdout, method.t_holdout, W)

                method.losses = np.hstack([method.losses, [[train_loss], [holdout_loss], [test_loss]]])
                method.errors = np.hstack([method.errors, [[train_error], [holdout_error], [test_error]]])
                #print train_loss, train_error, holdout_loss, holdout_error, test_loss, test_error
            else:
                method.losses = np.hstack([method.losses, [[train_loss], [test_loss]]])
                method.errors = np.hstack([method.errors, [[train_error], [test_error]]])
                #print train_loss, train_error, test_loss, test_error

            W += learning_rate * dW
            start, end = start + batch_size, end + batch_size if j != mini_batch - 1 else n

        if early_stop:
            if i != 0 and method.errors[1, -1] >= method.errors[1, -1 - mini_batch]:
                up_epoch += 1
                if up_epoch == early_stop:
                    idx = method.errors[1, :].argmin()
                    return weights[idx]
            else:
                up_epoch = 0

    if early_stop:
        idx = method.errors[1, :].argmin()
        W = weights[idx]
    return W


# Nesterov Momentum
def nesterov_momentum(method, W, learning_rate, mu, anneal, early_stop, mini_batch):
    X, t = method.X, method.t
    n = t.size
    batch_size = n / mini_batch
    v = 0
    weights = []
    for i in xrange(200):
        start, end = 0, batch_size
        for j in xrange(mini_batch):
            X_batch, t_batch = X[:, start:end], t[start:end]

            train_loss, train_error = method.eval_loss_and_error(X, t, W)
            test_loss, test_error = method.test(test_images, test_labels, W)

            dW = method.eval_gradient(X_batch, t_batch, W)

            learning_rate = learning_rate if not anneal else learning_rate / (1. + i / anneal) if i > 5 else 1e-1

            weights.append(np.array(W))

            if early_stop:
                holdout_loss, holdout_error = method.eval_loss_and_error(method.X_holdout, method.t_holdout, W)

                method.losses = np.hstack([method.losses, [[train_loss], [holdout_loss], [test_loss]]])
                method.errors = np.hstack([method.errors, [[train_error], [holdout_error], [test_error]]])
                #print train_loss, train_error, holdout_loss, holdout_error, test_loss, test_error
            else:
                method.losses = np.hstack([method.losses, [[train_loss], [test_loss]]])
                method.errors = np.hstack([method.errors, [[train_error], [test_error]]])
                #print train_loss, train_error, test_loss, test_error

            v_prev = v
            v = mu * v - learning_rate * dW
            W -= -mu * v_prev + (1 + mu) * v

            start, end = start + batch_size, end + batch_size if j != mini_batch - 1 else n

        if i != 0 and method.errors[1, -1] >= method.errors[1, -1 - mini_batch]:
            up_epoch += 1
            if up_epoch == early_stop:
                idx = method.errors[1, :].argmin()
                return weights[idx]
        else:
            up_epoch = 0

    if early_stop:
        idx = method.errors[1, :].argmin()
        W = weights[idx]

    return W


def plot_losses(method):
    fig, ax = plt.subplots()
    ax.plot(range(method.losses[0].size), method.losses[0], label="train loss")
    if method.early_stop:
        ax.plot(range(method.losses[1].size), method.losses[1], label="hold out loss")
        ax.plot(range(method.losses[2].size), method.losses[2], label="test loss")
    else:
        ax.plot(range(method.losses[1].size), method.losses[1], label="test loss")
    ax.legend()


def plot_errors(method):
    fig, ax = plt.subplots()
    ax.plot(range(method.errors[0].size), 1 - method.errors[0], label="train percent correct")
    if method.early_stop:
        ax.plot(range(method.errors[1].size), 1 - method.errors[1], label="hold out percent correct")
        ax.plot(range(method.errors[2].size), 1 - method.errors[2], label="test percent correct")
    else:
        ax.plot(range(method.errors[1].size), 1 - method.errors[1], label="test percent correct")
    ax.legend(loc="lower right")


def plot_weight(weight):
    weight_graph = weight[:, :-1].reshape(28, 28)
    fig, ax = plt.subplots()
    im = ax.imshow(weight_graph)
    fig.colorbar(im)


# class of logistic
class Logistic:
    # pick out expected numbers
    @staticmethod
    def select_numbers(images, labels, num1, num2):
        mask = np.any([labels==num1, labels==num2], axis=0)
        images, labels = images[:, mask], labels[mask]
        labels[labels==num1], labels[labels==num2] = 1, 0
        return images, labels

    # initialization
    # fit the model using gradient descent
    def __init__(self, train_images, train_labels, num1=2, num2=3, method="GD", learning_rate=1.,
                 anneal=0., early_stop=0, mini_batch=1, reg_type="", reg_weight=0., momentum=0.9):
        self.num1, self.num2 = num1, num2
        self.X, self.t = self.select_numbers(train_images, train_labels, num1, num2)
        self.X, self.t = shuffle(self.X, self.t)
        self.X = insert_bias(whitening(self.X))
        self.early_stop = early_stop
        if early_stop:
            holdout_size = self.X.shape[1] / 10
            self.X_holdout, self.t_holdout = self.X[:, :holdout_size], self.t[:holdout_size]
            self.X, self.t = self.X[:, holdout_size:], self.t[holdout_size:]
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.losses = np.array([[], [], []]) if early_stop else np.array([[], []])
        self.errors = np.array([[], [], []]) if early_stop else np.array([[], []])
        w0 = 1e-3 * np.random.randn(1, self.X.shape[0])
        if method == "GD":
            self.weight = gradient_descent(self, w0, learning_rate, anneal, early_stop, mini_batch)
        elif method == "NAG":
            self.weight = nesterov_momentum(self, w0, learning_rate, momentum, anneal, early_stop, mini_batch)

    def eval_loss_and_error(self, X, t, W):
        y = 1 / (1 + np.exp(-np.dot(W, X)))
        loss = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) / t.size
        if self.reg_type == "L2":
            loss += np.sum(W * W) * self.reg_weight
        elif self.reg_type == "L1":
            loss += np.sum(np.abs(W)) * self.reg_weight
        error = np.mean(np.abs((y > 0.5) - t))
        return loss, error

    def eval_gradient(self, X, t, W):
        y = 1 / (1 + np.exp(-np.dot(W, X)))
        gradient = np.dot(t - y, X.T) / t.size
        if self.reg_type == "L2":
            gradient -= self.reg_weight * W * 2
        elif self.reg_type == "L1":
            reg_grad = np.array(W)
            reg_grad[reg_grad>0] = 1
            reg_grad[reg_grad<0] = -1
            gradient -= self.reg_weight * reg_grad
        return gradient

    # test on the test set, if weight not defined then use self.weight
    def test(self, X_test, t_test, weight=0):
        X_test, t_test = self.select_numbers(test_images, test_labels, self.num1, self.num2)
        X_test = insert_bias(whitening(X_test))
        if type(weight) is int:
            return self.eval_loss_and_error(X_test, t_test, self.weight)
        else:
            return self.eval_loss_and_error(X_test, t_test, weight)

    def plot_weight(self):
        plot_weight(self.weight)

    def plot_losses(self):
        plot_losses(self)

    def plot_errors(self):
        plot_errors(self)


class Softmax():
    def __init__(self, train_images, train_labels, nClasses=10, learning_rate=1., anneal=0., early_stop=5,
                 mini_batch=1, reg_type="", reg_weight=0., method="GD", momentum=0.9):
        self.X, self.t = shuffle(train_images, train_labels)
        self.X = whitening(self.X)
        self.X = insert_bias(self.X)
        self.early_stop = early_stop
        if early_stop:
            holdout_size = self.X.shape[1] / 10
            self.X_holdout, self.t_holdout = self.X[:, :holdout_size], self.t[:holdout_size]
            self.X, self.t = self.X[:, holdout_size:], self.t[holdout_size:]
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.losses = np.array([[], [], []]) if early_stop else np.array([[], []])
        self.errors = np.array([[], [], []]) if early_stop else np.array([[], []])
        w0 = 1e-3 * np.random.randn(nClasses, self.X.shape[0])
        if method == "GD":
            self.weight = gradient_descent(self, w0, learning_rate, anneal, early_stop, mini_batch)
        elif method == "NAG":
            self.weight = nesterov_momentum(self, w0, learning_rate, momentum, anneal, early_stop, mini_batch)

    def eval_loss_and_error(self, X, t, W):
        n = t.size
        scores = np.exp(np.dot(W, X))
        probs = scores / np.sum(scores, axis=0, keepdims=True)
        loss = -np.sum(np.log(probs[t, range(n)])) / n
        if self.reg_type == "L2":
            loss += np.sum(W * W) * self.reg_weight
        elif self.reg_type == "L1":
            loss += np.sum(np.abs(W)) * self.reg_weight
        predict = probs.argmax(axis=0)
        error = np.mean(predict!=t)
        return loss, error

    def eval_gradient(self, X, t, W):
        n = t.size
        scores = np.exp(np.dot(W, X))
        probs = scores / np.sum(scores, axis=0, keepdims=True)
        probs[t, range(n)] -= 1
        gradient = -np.dot(probs, X.T) / n
        if self.reg_type == "L2":
            gradient -= 2 * W * self.reg_weight
        elif self.reg_type == "L1":
            reg_grad = np.array(W)
            reg_grad[reg_grad>0] = 1
            reg_grad[reg_grad<0] = -1
            gradient -= self.reg_weight * reg_grad
        return gradient

    # test on the test set, if weight not defined then use self.weight
    def test(self, X_test, t_test, weight=0):
        X_test = insert_bias(whitening(X_test))
        if type(weight) is int:
            return self.eval_loss_and_error(X_test, t_test, self.weight)
        else:
            return self.eval_loss_and_error(X_test, t_test, weight)

    def plot_losses(self):
        plot_losses(self)

    def plot_errors(self):
        plot_errors(self)

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_images()

    # Batch gradient descent
    logistic = Logistic(train_images, train_labels, method="GD",
                        learning_rate=1e0, anneal=1e0, early_stop=3)
    print "Loss and error of batch gradient descent: ", logistic.test(test_images, test_labels)
    logistic.plot_losses()
    logistic.plot_errors()
    logistic.plot_weight()

    # Mini-batch gradient descent
    logistic = Logistic(train_images, train_labels, method="GD",
                        learning_rate=1e0, anneal=1e0, early_stop=3, mini_batch=10)
    print "Loss and error of mini-batch gradient descent: ", logistic.test(test_images, test_labels)
    logistic.plot_losses()
    logistic.plot_errors()
    logistic.plot_weight()

    # Nesterov momentum
    logistic = Logistic(train_images, train_labels, method="NAG", momentum=0.9,
                        learning_rate=1e-1, early_stop=50, mini_batch=10)
    print "Loss and error of Nesterov momentum: ", logistic.test(test_images, test_labels)
    logistic.plot_losses()
    logistic.plot_errors()
    logistic.plot_weight()

    # Difference between 2 vs 3 and 2 vs 8
    logistic3 = Logistic(train_images, train_labels, num2=3, method="NAG", momentum=0.9,
                         learning_rate=1e-1, early_stop=5, mini_batch=10)
    logistic8 = Logistic(train_images, train_labels, num2=8, method="NAG", momentum=0.9,
                         learning_rate=1e-1, early_stop=5, mini_batch=10)
    print "Loss and error of 2 vs 3 ", logistic3.test(test_images, test_labels)
    print "Loss and error of 2 vs 8 ", logistic8.test(test_images, test_labels)
    logistic3.plot_weight()
    logistic8.plot_weight()

    # Comparison with 3 vs 8
    wd = logistic8.weight - logistic3.weight
    plot_weight(wd)
    logistic38 = Logistic(train_images, train_labels, num1=3, num2=8, method="NAG", momentum=0.9,
                          learning_rate=1e-1, early_stop=5, mini_batch=10)
    logistic38.plot_weight()
    print "Loss and error of using the difference weight of classifier 2 vs 3 and 2 vs 8 ",\
        logistic38.test(test_images, test_labels, wd)
    print "Loss and error of 3 vs 8 ", logistic38.test(test_images, test_labels)

    # Regularization
    def regularization(lambdas, type):
        train_errors, weight_length, test_errors = np.array([]), np.array([]), np.array([])

        for l in lambdas:
            logistic = Logistic(train_images, train_labels, method="NAG", momentum=0.9, early_stop=0,
                                learning_rate=1e-1, reg_type=type, reg_weight=l)
            logistic.plot_weight()

            train_errors = np.hstack([train_errors, logistic.errors[0, -1]])
            weight_length = np.hstack([weight_length, np.sum((logistic.weight) ** 2)])
            _, test_error = logistic.test(test_images, test_labels)
            test_errors = np.hstack([test_errors, test_error])

        def plots(v, label):
            fig, ax = plt.subplots()
            ax.plot(lambdas, v, label=label)
            ax.set_xscale("log")
            ax.legend()

        plots(1 - train_errors, "train percent correct")
        plots(1 - test_errors, "test percent correct")
        plots(weight_length, "weight length")

    lambdas = [10 ** i for i in xrange(-6, 0)]
    regularization(lambdas, "L1")

    lambdas = [10 ** i for i in xrange(-6, 1)]
    regularization(lambdas, "L2")

    # Softmax
    softmax = Softmax(train_images, train_labels, method="NAG", learning_rate=1e0, momentum=0.9,
                      reg_type="L2", reg_weight=1e-3, mini_batch=10, early_stop=5)
    print "Loss and error of softmax classifier: ", softmax.test(test_images, test_labels)
    softmax.plot_losses()
    softmax.plot_errors()
    plt.show()
