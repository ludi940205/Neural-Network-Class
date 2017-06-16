from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt


def load_images():
    # load the MNIST data set
    mndata = MNIST('.')
    mndata.load_training()
    mndata.load_testing()

    # using numpy array form to store the train and test sets
    # and transform the data type to double
    train_images = np.array(mndata.train_images).T / 255.
    train_labels = np.array(mndata.train_labels).T
    test_images = np.array(mndata.test_images).T / 255.
    test_labels = np.array(mndata.test_labels).T
    return train_images, train_labels, test_images, test_labels


# pre-process the data so that each pixel will have roughly zero mean and unit variance.
def whitening(images):
    m = np.mean(images, axis=0)
    images -= m
    return images


# insert an extra dimension for the bias term
def insert_bias(X):
    X = np.insert(X, X.shape[0], 1, axis=0)
    return X


# randomly shuffle the data
def shuffle(X, t):
    idx = np.random.permutation(t.size)
    X, t = X[:, idx], t[idx]
    return X, t


def plot_losses(method):
    fig, ax = plt.subplots()
    niter = method.losses.shape[1]
    x = np.linspace(1, niter / method.mini_batch, niter)
    ax.plot(x, method.losses[0], label="train loss")
    if method.early_stop:
        ax.plot(x, method.losses[1], label="validation loss")
        ax.plot(x, method.losses[2], label="test loss")
    else:
        ax.plot(x, method.losses[1], label="test loss")
    ax.legend()


def plot_errors(method):
    fig, ax = plt.subplots()
    niter = method.errors.shape[1]
    x = np.linspace(1, niter / method.mini_batch, niter)
    ax.plot(x, 1 - method.errors[0], label="train percent correct")
    if method.early_stop:
        ax.plot(x, 1 - method.errors[1], label="validation percent correct")
        ax.plot(x, 1 - method.errors[2], label="test percent correct")
    else:
        ax.plot(x, 1 - method.errors[1], label="test percent correct")
    ax.legend(loc="lower right")


def sigmoid(X, W):
    return 1 / (1 + np.exp(np.dot(W.T, X)))


def softmax(X, W):
    ak = np.dot(W.T, X)
    scores = np.exp(ak)
    return scores / np.sum(scores, axis=0, keepdims=True)


def tanh_sigmoid(X, W):
    a = np.dot(W.T, X)
    return 1.7159 * np.tanh(2.0 / 3.0 * a)


class MultilayerClassifier():
    def __init__(self, train_images, train_labels, nClasses=10, nHiddenUnits=(100,), shuffle=True,
                 learning_rate=1., anneal=0., early_stop=3, mini_batch=1, reg_type="", reg_weight=0., method="GD",
                 momentum=0.9, act_funs=("sigmoid", "softmax"), custom_init_weight=0.):
        self.X, self.t = train_images, train_labels
        self.X = whitening(self.X)
        self.X = insert_bias(self.X)
        self.early_stop = early_stop
        if early_stop:
            validation_size = self.X.shape[1] / 6
            self.X_validation, self.t_validation = self.X[:, :validation_size], self.t[:validation_size]
            self.X, self.t = self.X[:, validation_size:], self.t[validation_size:]
        self.act_funs = act_funs
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.mini_batch = mini_batch
        self.shuffle = shuffle
        self.losses = np.array([[], [], []]) if early_stop else np.array([[], []])
        self.errors = np.array([[], [], []]) if early_stop else np.array([[], []])
        nLayer = len(nHiddenUnits) + 1
        nUnits = (self.X.shape[0],) + nHiddenUnits + (nClasses,)
        if not custom_init_weight:
            self.weights = [1 / np.sqrt(nUnits[i]) * np.random.randn(nUnits[i], nUnits[i + 1]) for i in xrange(nLayer)]
        else:
            self.weights = [custom_init_weight * np.random.randn(nUnits[i], nUnits[i + 1]) for i in xrange(nLayer)]
        if method:
            self.weights = self.optimization(self.weights, learning_rate, anneal, early_stop, mini_batch, method, momentum)

    def optimization(self, weights, learning_rate, anneal, early_stop, mini_batch, method, mu):
        d, n = self.X.shape
        batch_size = n / mini_batch
        weights_records = [[] for _ in weights]
        velocities = [0 for _ in weights]
        train_loss = train_error = validation_loss = validation_error = test_loss = test_error = 0
        for i in xrange(2000):
            start, end = 0, batch_size
            if self.shuffle:
                self.X, self.t = shuffle(self.X, self.t)
            for j in xrange(mini_batch):
                X_batch, t_batch = self.X[:, start:end], self.t[start:end]

                lr = learning_rate if not anneal else \
                    learning_rate / (1. + i / anneal) if i * mini_batch + j > 5 else 1e0

                outputs = self.forwardProp(X_batch, weights)
                dWeights = self.backProp(X_batch, t_batch, outputs, weights)
                train_loss, train_error = self.eval_loss_and_error(X_batch, t_batch, weights)

                if j % 100 == 0:
                    test_loss, test_error = self.test(test_images, test_labels, weights)

                for k in xrange(len(weights_records)):
                    weights_records[k].append(np.array(weights[k]))

                if early_stop:
                    if j % 100 == 0:
                        validation_loss, validation_error = self.eval_loss_and_error(self.X_validation, self.t_validation, weights)

                    self.losses = np.hstack([self.losses, [[train_loss], [validation_loss], [test_loss]]])
                    self.errors = np.hstack([self.errors, [[train_error], [validation_error], [test_error]]])
                    #print train_loss, train_error, validation_loss, validation_error, test_loss, test_error
                else:
                    self.losses = np.hstack([self.losses, [[train_loss], [test_loss]]])
                    self.errors = np.hstack([self.errors, [[train_error], [test_error]]])
                    #print train_loss, train_error, test_loss, test_error

                if method == "GD":
                    weights = map(lambda x, y: x - lr * y, weights, dWeights)
                elif method == "NAG":
                    for k in xrange(len(weights)):
                        v_prev = velocities[k]
                        velocities[k] = mu * velocities[k] - lr * dWeights[k]
                        weights[k] += -mu * v_prev + (1 + mu) * velocities[k]

                start, end = start + batch_size, end + batch_size if j != mini_batch - 1 else n

            #test_loss, test_error = self.test(test_images, test_labels, weights)
            if early_stop:
                #print train_loss, train_error, validation_loss, validation_error, test_loss, test_error
                if i != 0 and self.errors[1, -1] >= self.errors[1, -1 - mini_batch]:
                    up_epoch += 1
                    if up_epoch == early_stop:
                        idx = self.errors[1, :].argmin()
                        return [weights_record[idx] for weights_record in weights_records]
                else:
                    up_epoch = 0
            else:
                pass
                #print train_loss, train_error, test_loss, test_error

        if early_stop:
            idx = self.errors[1, :].argmin()
            weights = [weights_record[idx] for weights_record in weights_records]
        return weights

    def eval_loss_and_error(self, X, t, weights):
        outputs = self.forwardProp(X, weights)
        y = outputs[-1]
        n = t.size
        loss = -np.sum(np.log(y[t, range(n)])) / n

        for weight in weights:
            if self.reg_type == "L2":
                loss += (np.sum(weight * weight)) * self.reg_weight
            elif self.reg_type == "L1":
                loss += (np.sum(np.abs(weight))) * self.reg_weight

        predict = y.argmax(axis=0)
        error = np.mean(predict != t)
        return loss, error

    def forwardProp(self, X, weights):
        outputs = [X]
        for weight, act_fun in zip(weights, self.act_funs):
            if act_fun == "sigmoid":
                outputs.append(sigmoid(outputs[-1], weight))
            elif act_fun == "softmax":
                outputs.append(softmax(outputs[-1], weight))
            elif act_fun == "tanh":
                outputs.append(tanh_sigmoid(outputs[-1], weight))
        return outputs[1:]

    def backProp(self, X, t, outputs, weights):
        n = t.size
        dWeights = []
        outputs = [X] + outputs
        for i in xrange(len(weights) - 1, -1, -1):
            output1, output2 = outputs[i], outputs[i + 1]
            if self.act_funs[i] == "softmax":
                output2[t, range(n)] -= 1  # computing y - t
                delta = output2 / n
                dWeights.append(np.dot(output1, delta.T))
            elif self.act_funs[i] == "sigmoid":
                delta = -output2 * (1 - output2) * np.dot(weights[i + 1], delta)
                dWeights.append(np.dot(output1, delta.T))
            elif self.act_funs[i] == "tanh":
                f = output2 / 1.7159
                delta = 1.14393 * (1 - f ** 2) * np.dot(weights[i + 1], delta)
                dWeights.append(np.dot(output1, delta.T))
        dWeights = dWeights[::-1]

        for i in xrange(len(weights)):
            if self.reg_type == "L2":
                dWeights[i] += 2 * weights[i] * self.reg_weight
            elif self.reg_type == "L1":
                regW = np.array(weights[i])
                regW[regW>0] = 1
                regW[regW<0] = -1
                dWeights[i] += self.reg_weight * regW

        return dWeights

    # test on the test set, if weight not defined then use self.weight
    def test(self, X_test, t_test, weights=0):
        #X_test = insert_bias(whitening(X_test))

        if type(weights) is int:
            return self.eval_loss_and_error(X_test, t_test, self.weights)
        else:
            return self.eval_loss_and_error(X_test, t_test, weights)

    def plot_losses(self):
        plot_losses(self)

    def plot_errors(self):
        plot_errors(self)

if  __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_images()
    test_images = insert_bias(whitening(test_images))

    """
    Gradient check
    """

    def gradient_check(dW):
        multilayer = MultilayerClassifier(train_images, train_labels, nHiddenUnits=(2,),
                                      reg_type="L2", reg_weight=1e-3, method="", act_funs=("tanh", "softmax"))
        X, weights = np.random.randn(785, len(multilayer.weights)), []
        for weight in multilayer.weights:
            weights.append(np.random.randn(weight.shape[0], weight.shape[1]))
        weights = multilayer.weights
        t = np.random.randint(0, 9, len(multilayer.weights))
        outputs = multilayer.forwardProp(X, weights)
        analytic = multilayer.backProp(X, t, outputs, weights)
        numeric = [np.zeros(weight.shape) for weight in weights]

        for k in xrange(len(numeric)):
            for i in xrange(weights[k].shape[0]):
                for j in xrange(weights[k].shape[1]):
                    W_positive, W_negative = np.array(weights[k]), np.array(weights[k])
                    W_positive[i, j] += dW
                    W_negative[i, j] -= dW
                    loss_positive, _ = multilayer.eval_loss_and_error(X, t, weights[:k] + [W_positive] + weights[k+1:])
                    loss_negative, _ = multilayer.eval_loss_and_error(X, t, weights[:k] + [W_negative] + weights[k+1:])
                    numeric[k][i, j] = (loss_positive - loss_negative) / (2 * dW)

        for i, (num, ana) in enumerate(zip(numeric, analytic)):
            print "Layer %d: maximum difference between numeric and analytic gradient is %e." % (i, np.max(abs(ana - num)))

    gradient_check(1e-2)

    """
    Trick of the Trade
    """

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e0, early_stop=5, method="GD",
                                      reg_type='L2', reg_weight=1e-4, shuffle=False, custom_init_weight=0.001)
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for batch gradient descent is ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e0, early_stop=3, method="GD",
                                      mini_batch=400, reg_type='L2', reg_weight=1e-5, shuffle=False, custom_init_weight=0.001)
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for minibatch gradient descent is ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e0, early_stop=3, method="GD",
                                      mini_batch=400, reg_type='L2', reg_weight=1e-5, custom_init_weight=0.001)
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for shuffle ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e0, early_stop=3, method="GD",
                                      act_funs=('tanh', 'softmax'), mini_batch=400, reg_type='L2', reg_weight=1e-5,
                                      custom_init_weight=0.001)
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for tanh sigmoid ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e0, early_stop=3, method="GD",
                                      act_funs=('tanh', 'softmax'), mini_batch=400, reg_type='L2', reg_weight=1e-5)
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for initial weight ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e-1, early_stop=3, method="NAG",
                                      act_funs=('tanh','softmax'), reg_type='L2', reg_weight=0, mini_batch=400)
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for Nesterov momentum ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    """
    Topology
    """

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e-1, early_stop=3, method="NAG",
                                      act_funs=('tanh','softmax'), reg_type='L2', reg_weight=0, mini_batch=400,
                                      nHiddenUnits=(20, ))
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for less hidden units ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e-1, early_stop=3, method="NAG",
                                      act_funs=('tanh','softmax'), reg_type='L2', reg_weight=0, mini_batch=400,
                                      nHiddenUnits=(1000, ))
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for more hidden units ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()

    multilayer = MultilayerClassifier(train_images, train_labels, learning_rate=1e-1, early_stop=3, method="NAG",
                                      act_funs=('tanh', 'tanh','softmax'), reg_type='L2', reg_weight=0, mini_batch=400,
                                      nHiddenUnits=(50, 50))
    test_loss, test_error = multilayer.test(test_images, test_labels)
    print "Test error for 2 hidden layer network ", 1 - test_error
    multilayer.plot_losses()
    multilayer.plot_errors()
    plt.show()
