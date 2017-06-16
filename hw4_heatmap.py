from keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt


class Coding:
    def __init__(self, data):
        self.chars = sorted(set(data))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        X = np.zeros((len(s), len(self.chars)))
        for i, c in enumerate(s):
            X[i, self.char_to_idx[c]] = 1
        return X

    def decode(self, X):
        return ''.join(self.idx_to_char[x] for x in X)


# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 30  # number of steps to unroll the RNN for

data = open('data/input.txt', 'r').read()
size = len(data)
ctable = Coding(data)
vocab_size = len(ctable.chars)

model = load_model('models/model_hidden_size_100RMSprop')
hidden_layer = Model(input=model.input, output=model.get_layer(index=1).output)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    idx = np.random.choice(range(vocab_size), p=preds)
    return idx


"""
Generate music, record the activations of hidden units.
"""
prime = "d G2 G|cB A2|FG Bc|d2 z2|\nG2 G"
model = load_model('models/model_hidden_size_100RMSprop')
row = col = 30
len_gen = row * col
heatmap = np.zeros((hidden_size, row, col))
charmap = [['' for _ in range(col)] for _ in range(row)]

result = []
seed = prime
for i in range(len_gen):
    x = np.expand_dims(ctable.encode(seed), axis=0)
    preds = model.predict(x)[0][-1]
    hidden_activation = hidden_layer.predict(x)[0, -1, :]
    for j, h in enumerate(hidden_activation):
        heatmap[j, i // col, i % col] = h

    idx = sample(preds)
    next_char = ctable.decode([idx])
    if len(seed) >= seq_length:
        seed = seed[1:] + next_char
    else:
        seed += next_char
    result.append(next_char)
    charmap[i // col][i % col] = next_char

"""
Save generated musci to file
"""
gen_result = "".join(result)
file = open('heatmaps/generated_text.txt', 'w')
file.write(gen_result)
file.close()


"""
Plot heatmaps of all the hidden units and save the figures to file.
"""
for k in range(hidden_size):
    print(k)
    plt.figure(figsize=(20, 20))
    plt.imshow(heatmap[k, ...], cmap="seismic")
    for i in range(30):
        for j in range(30):
            c = gen_result[i*30+j]
            offset = 0.2 if c == ' ' or c == '\n' else 0
            c = 'sp' if c == ' ' else 'nl' if c == '\n' else c
            plt.text(j - offset - 0.1, i + 0.1, c, size='xx-large')
    plt.colorbar()
    plt.savefig('C:\\Users\\dilu\private\\hw4\\heatmaps\\heapmap' + str(k))
