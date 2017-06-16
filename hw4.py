from keras.layers import SimpleRNN, Dense, Activation
from keras.models import Sequential
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt


class Coding:
    def __init__(self, data):
        self.chars = sorted(set(data))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        """
        Input the string, output one hot coding
        :param s: the input string
        :return: one hot coding of the input string
        """
        X = np.zeros((len(s), len(self.chars)))
        for i, c in enumerate(s):
            X[i, self.char_to_idx[c]] = 1
        return X

    def decode(self, X):
        """
        Decode the list of indexes and return the corresponding string
        :param X: the list of indexes of a string
        :return: corresponding string
        """
        return ''.join(self.idx_to_char[x] for x in X)


class History(Callback):
    """
    used to record training loss and accuracy.
    """
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 30  # number of steps to unroll the RNN for

# read in data and encoding
data = open('data/input.txt', 'r').read()
size = len(data)
ctable = Coding(data)
vocab_size = len(ctable.chars)

val_size = int(size * 0.2)
train_data, val_data = data[val_size:], data[:val_size]
nb_train_samples, nb_val_samples = len(train_data) - seq_length + 1, len(val_data) - seq_length + 1


def data_gen(data, batch_size, nb_samples):
    """
    random generator of training and validation sequence
    :param data: the whole train or validation string
    :param batch_size: batch size
    :param nb_samples: number of samples per epoch
    :return: Random sequence in the input data with one hot coding and size batch_size.
    """
    n = 0
    while True:
        curr_size = batch_size
        if n + curr_size >= nb_samples:
            curr_size = nb_samples - n
            n = 0
        X_batch = np.zeros((curr_size, seq_length, vocab_size))
        y_batch = np.zeros((curr_size, seq_length, vocab_size))
        for i in range(curr_size):
            idx = np.random.randint(len(data) - seq_length - 1)
            input_str = data[idx:idx + seq_length]
            target_str = data[idx + 1:idx + seq_length + 1]
            X_batch[i, :, :] = ctable.encode(input_str)
            y_batch[i, :, :] = ctable.encode(target_str)
        yield X_batch, y_batch


# Initialize the training and validation generator
val_size = int(len(data) * 0.2)
batch_size = 32

train_gen = data_gen(train_data, batch_size, nb_train_samples)
val_gen = data_gen(val_data, batch_size, nb_val_samples)


def get_model(hidden_size, optimizer, dropout):
    """
    Get the RNN model.
    :param hidden_size: number of units in the hidden layer
    :param optimizer: optimizer
    :param dropout: the probability of dropout
    :return: The compiled RNN model
    """
    model = Sequential()
    model.add(SimpleRNN(hidden_size, input_shape=(None, vocab_size), return_sequences=True,
                        dropout_W=dropout, dropout_U=dropout))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())
    return model


def sample(preds, temperature=1.0):
    """
    Function to sample an index from a probability array
    :param preds: The probability distribution
    :param temperature: temperature parameter
    :return: the generated next index
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    idx = np.random.choice(range(vocab_size), p=preds)
    return idx


def plot_loss_and_error(history, filename):
    """
    plot loss and error v.s. iteration
    :param history: recorded history containing train and validation loss and accuracy
    :param filename: history file name
    :return: nothing
    """
    nb_epoch, nb_iter = len(history.val_losses), len(history.losses)
    t1 = np.linspace(0, nb_epoch, nb_iter)
    t2 = np.linspace(0, nb_epoch, nb_epoch)

    plt.figure()
    plt.plot(t1, history.losses, label="train loss")
    plt.plot(t2, history.val_losses, label="validation loss")
    plt.legend()
    plt.savefig('C:\\Users\\dilu\private\\hw4\\figures\\' + filename + 'loss')

    plt.figure()
    plt.plot(t1, history.acc, label="train accuracy")
    plt.plot(t2, history.val_acc, label="validation accuracy")
    plt.legend()
    plt.savefig('C:\\Users\\dilu\private\\hw4\\figures\\' + filename + 'acc')


"""
Training the model, save the model and history to file, plot loss and error curves,
and finally generate music.
Iterater through number of hidden units and temperature.
"""
for hidden_size in [50, 75, 100, 150]:
    for optimizer in ['Adagrad', 'RMSprop']:
        model = get_model(hidden_size, optimizer, 0)
        history = History()

        model.fit_generator(train_gen,
                            samples_per_epoch=nb_train_samples,
                            validation_data=val_gen,
                            nb_val_samples=nb_val_samples,
                            nb_epoch=1,
                            callbacks=[history],)

        model.save("models/model_hidden_size_" + str(hidden_size) + optimizer)
        np.save("history/history_" + str(hidden_size) + optimizer,
                [history.losses, history.acc, history.val_losses, history.val_acc])
        plot_loss_and_error(history, 'hidden' + str(hidden_size) + optimizer)

        for temperature in [0.5, 1, 2]:
            prime = "<start>\nX:"
            len_gen = 500
            gen_result = ""
            for j in range(3):
                result = []
                seed = prime
                for i in range(len_gen):
                    x = np.expand_dims(ctable.encode(seed), axis=0)
                    preds = model.predict(x)[0][-1]
                    idx = sample(preds, temperature=temperature)
                    next_char = ctable.decode([idx])
                    if len(seed) >= seq_length:
                        seed = seed[1:] + next_char
                    else:
                        seed = seed + next_char
                    result.append(next_char)
                gen_result += 'PIECE ' + str(j) + '\n\n' + prime + "".join(result) + '\n\n'
            print(gen_result)
            file = open('generated/hidden_' + str(hidden_size) + 't' + str(temperature) + optimizer + '.txt', 'w')
            file.write(gen_result)
            file.close()


"""
Iterate through dropout probability.
"""
for dropout in [0.1, 0.2, 0.3]:
    optimizer = "RMSprop"
    suffix1 = "dropout_0_" + str(int(10 * dropout))
    suffix2 = "dropout_" + str(dropout)
    model = get_model(hidden_size, optimizer, dropout)
    history = History()

    model.fit_generator(train_gen,
                        samples_per_epoch=nb_train_samples,
                        validation_data=val_gen,
                        nb_val_samples=nb_val_samples,
                        nb_epoch=15,
                        callbacks=[history],)

    model.save("models/model_" + suffix2)
    np.save("history/history_" + suffix2, [history.losses, history.acc, history.val_losses, history.val_acc])
    plot_loss_and_error(history, suffix1)

    for temperature in [0.5, 1, 2]:
        prime = "<start>\nX:"
        len_gen = 500
        gen_result = ""
        for j in range(3):
            result = []
            seed = prime
            for i in range(len_gen):
                x = np.expand_dims(ctable.encode(seed), axis=0)
                preds = model.predict(x)[0][-1]
                idx = sample(preds, temperature=temperature)
                next_char = ctable.decode([idx])
                if len(seed) >= seq_length:
                    seed = seed[1:] + next_char
                else:
                    seed = seed + next_char
                result.append(next_char)
            gen_result += 'PIECE ' + str(j + 1) + '\n\n' + prime + "".join(result) + '\n\n'
        file = open('generated/dropout_' + str(dropout) + 't' + str(temperature) + '.txt', 'w')
        file.write(gen_result)
        file.close()

plt.show()
