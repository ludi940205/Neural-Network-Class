import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications import VGG16
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Lambda
from keras import optimizers
import glob
from keras.preprocessing import image
from keras.utils import np_utils
from keras.callbacks import Callback, TensorBoard, EarlyStopping
from keras import backend as K
import shutil
import os
import random
from keras.preprocessing.image import ImageDataGenerator
import time


class History(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))


class TempSoftmax(Dense):
    def __init__(self, output_dim, T, **kwargs):
        self.T = T
        super(TempSoftmax, self).__init__(output_dim, **kwargs)

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        output /= self.T
        return self.activation(output)


def shuffle(X, y):
    perm = np.random.permutation(len(y))
    X, y = np.array(X)[perm], np.array(y)[perm]
    return X, y


def plotLoss(model):
    fig, ax = plt.subplots()
    nb_epoch, nb_iter = len(model.history.val_losses), len(model.history.losses)
    t1 = np.linspace(1, nb_epoch, nb_iter)
    t2 = np.linspace(1, nb_epoch, nb_epoch)
    ax.plot(t1, model.history.losses, label="train loss")
    ax.plot(t2, model.history.val_losses, label="validation loss")
    ax.legend()


def plotAccuracy(model):
    fig, ax = plt.subplots()
    nb_epoch, nb_iter = len(model.history.val_acc), len(model.history.acc)
    t1 = np.linspace(1, nb_epoch, nb_iter)
    t2 = np.linspace(1, nb_epoch, nb_epoch)
    ax.plot(t1, model.history.acc, label="train accuracy")
    ax.plot(t2, model.history.val_acc, label="validation accuracy")
    ax.legend()


def plotWeight(model, layer_name, img_path):
    img = image.load_img(img_path, target_size=(model.image_size[0], model.image_size[1]))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(x)
    intermediate_layer_model = Model(input=model.model.input,
                                     output=model.model.get_layer(name=layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x)

    n = intermediate_output.shape[-1]
    row = col = np.sqrt(n)
    if row % 1 != 0:
        row = np.sqrt(n / 2)
        col = 2 * row
    row, col = int(row), int(col)
    fig, ax = plt.subplots(row, col)
    for i in range(n):
        r, c = i / col, i % col
        ax[r, c].imshow(intermediate_output[0, :, :, i])
        ax[r, c].axis('off')


def fileSplitter(path):
    filenames = [f for f in glob.glob(path + "/*.jpg")]
    categories = {}
    temp = 0
    for i, filename in enumerate(filenames):
        category = filename.split('\\')[-1].split('_')[0]
        if category not in categories:
            categories[category] = {"index": temp, "filenames": []}
            temp += 1
        categories[category]['filenames'].append(filename)

    return categories


class VGG16TransferModel():
    def __init__(self, path, load_exist_model=False, **kwargs):
        self.params = kwargs
        self.image_size = (224, 224, 3)

        if len(glob.glob(path + '/*/')) == 0:
            data = fileSplitter(path)
            self.nb_class = len(data)
            train_x, train_y, valid_x, valid_y = self.selectSamples(data)
            self.nb_train_img, self.nb_test_img = train_x.shape[0], valid_x.shape[0]
            self.train_generator = self.imageGenerator(train_x, train_y, self.params["batch_size"])
            self.valid_generator = self.imageGenerator(valid_x, valid_y, 16)
        else:
            self.nb_train_img, self.nb_test_img = self.moveFilesForVal()
            datagen = ImageDataGenerator()
            self.train_generator = self.preprocess(datagen.flow_from_directory(
                './data/temp/trainImg/',
                target_size=(224, 224),
                batch_size=params["batch_size"]
            ))
            self.valid_generator = self.preprocess(datagen.flow_from_directory(
                './data/temp/testImg/',
                target_size=(224, 224),
                batch_size=params["batch_size"]
            ))
            self.nb_class = 257

        if load_exist_model:
            self.model = load_model(load_exist_model)
        else:
            self.model = self.getModel()

        self.history = History()
        self.train_hist = self.train()

    def train(self):
        """
        compile and train the model.

        :param kwargs:
        :return:
        """
        if self.params["early_stop"]:
            early_stopping = EarlyStopping(patience=self.params["early_stop"])
            return self.model.fit_generator(
                self.train_generator,
                samples_per_epoch=self.nb_train_img,
                validation_data=self.valid_generator,
                nb_val_samples=self.nb_test_img,
                nb_epoch=self.params["nb_epoch"],
                callbacks=[self.history, TensorBoard(), early_stopping]
            )
        else:
            return self.model.fit_generator(
                self.train_generator,
                samples_per_epoch=self.nb_train_img,
                validation_data=self.valid_generator,
                nb_val_samples=self.nb_test_img,
                nb_epoch=self.params["nb_epoch"],
                callbacks=[self.history, TensorBoard()]
            )

    def getModel(self):
        """
        Set and compile the model.

        :return: compiled model
        """
        vgg_model = VGG16(weights='imagenet', include_top=True)
        if self.params['T'] == 0:
            vgg_out = vgg_model.get_layer(name=self.params["intermed_layer"]).output
            if vgg_out.get_shape().ndims > 2:
                vgg_out = Flatten(name='flatten')(vgg_out)
            softmax_layer = Dense(self.nb_class, activation='softmax', name='my_predictions')(vgg_out)
            tl_model = Model(input=vgg_model.input, output=softmax_layer)
        else:
            weights = vgg_model.get_weights()
            weights_len = len(weights)

            vgg_output_layer = vgg_model.get_layer(name="fc2")
            vgg_out = vgg_output_layer.output
            temp_softmax_layer = TempSoftmax(1000, T=self.params['T'], activation='softmax', name='temperature_based')(
                vgg_out)
            output_layer = Dense(self.nb_class,activation='softmax',name='final_output')(temp_softmax_layer)
            tl_model = Model(input=vgg_model.input, output=output_layer)

            new_weights = tl_model.get_weights()
            new_weights[0:weights_len] = weights
            tl_model.set_weights(new_weights)

        # Freeze all layers of VGG16 and Compile the model
        for layer in tl_model.layers[1:-1]:
            layer.trainable = False

        if self.params["optimizer"] == 'SGD':
            tl_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.SGD(lr=self.params['lr'], momentum=self.params['momentum']),
                             metrics=['categorical_accuracy'])
        elif self.params["optimizer"] == "RMSprop":
            tl_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.RMSprop(lr=self.params['lr']),
                             metrics=['categorical_accuracy'])
        elif self.params["optimizer"] == "Adagrad":
            tl_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.Adagrad(lr=self.params['lr']),
                             metrics=['categorical_accuracy'])
        elif self.params["optimizer"] == "Adadelta":
            tl_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.Adadelta(lr=self.params['lr']),
                             metrics=['categorical_accuracy'])
        elif self.params["optimizer"] == "Adam":
            tl_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.Adam(lr=self.params['lr']),
                             metrics=['categorical_accuracy'])

        return tl_model

    def printModelArchitecture(self):
        # Confirm the model is appropriate
        print(self.model.summary())

    def selectSamples(self, data):
        """
        Select some number of training examples from every category. Split hold-out set from original training set.

        :param categories: a dictionary of which each entry contains one class of samples.
        :param nb_train_per_category: number of examples per category.
        :return:
        """
        train_x, train_y, valid_x, valid_y = [], [], [], []
        for category in data:
            filenames, index = data[category]["filenames"], data[category]["index"]
            np.random.shuffle(filenames)
            nb_valid = int(len(filenames) * self.params["valid_split"])
            train, valid = filenames[nb_valid:], filenames[:nb_valid]

            train_x.extend(train[:self.params["nb_per_class"]])
            train_y.extend([index] * self.params["nb_per_class"])
            valid_x.extend(valid)
            valid_y.extend([index] * nb_valid)

        train_x, train_y = shuffle(train_x, train_y)
        valid_x, valid_y = shuffle(valid_x, valid_y)

        train_y = np_utils.to_categorical(train_y, self.nb_class)
        valid_y = np_utils.to_categorical(valid_y, self.nb_class)

        return train_x, train_y, valid_x, valid_y

    def imageGenerator(self, filenames, targets, batch_size=0):
        """
        The generator of images which receives the file names.

        :param filenames: file names of train/validation samples.
        :param targets: true class of each samples.
        :param batch_size: batch size.
        :return: batch of images and target.
        """
        start, n = 0, targets.shape[0]
        if batch_size == 0:
            batch_size = n
        while True:
            if start == 0:
                index_array = np.random.permutation(n)

            end = min(start + batch_size, n)
            curr_batch_size = end - start

            batch_x = np.zeros((curr_batch_size,) + self.image_size)
            for i, j in enumerate(index_array[start: end]):
                img = image.load_img(filenames[j], target_size=(self.image_size[0], self.image_size[1]))
                x = image.img_to_array(img)
                batch_x[i] = x
            batch_x = preprocess_input(batch_x)

            batch_y = np.zeros((len(batch_x), targets.shape[1]), dtype='float32')
            for i, j in enumerate(index_array[start: end]):
                batch_y[i] = targets[j]

            start = end if end < n else 0

            yield batch_x, batch_y

    def moveFilesForVal(self):
        print("Copying files...")
        valSize, nb_per_category = self.params["valid_split"], self.params["nb_per_class"]
        nb_train_img, nb_test_img = 0, 0
        path = "./data/256_ObjectCategories/"
        if os.path.exists("./data/temp/trainImg/"):
            shutil.rmtree("./data/temp/trainImg/")
        if os.path.exists("./data/temp/testImg/"):
            shutil.rmtree("./data/temp/testImg/")
        os.makedirs("./data/temp/trainImg/")
        os.makedirs("./data/temp/testImg/")
        for nb_cls, cls in enumerate(os.listdir(path)):
            # if nb_cls > 20:
            #     break
            if cls.startswith("."):
                continue
            imgFiles = os.listdir(path + cls)
            random.shuffle(imgFiles)
            if not os.path.exists("./data/temp/trainImg/" + cls):
                os.makedirs("./data/temp/trainImg/" + cls)
            if not os.path.exists("./data/temp/testImg/" + cls):
                os.makedirs("./data/temp/testImg/" + cls)

            for i, img in enumerate(imgFiles):
                if os.path.isdir(path + cls + "/" + img):
                    continue
                if i < nb_per_category:
                    shutil.copy(path + cls + "/" + img, "./data/temp/trainImg/" + cls)
                    nb_train_img += 1
                elif i < len(imgFiles) * valSize + nb_per_category - 1:
                    shutil.copy(path + cls + "/" + img, "./data/temp/testImg/" + cls)
                    nb_test_img += 1
        print("done.")

        return nb_train_img, nb_test_img

    def preprocess(self, gen):
        for img, label in gen:
            yield preprocess_input(img), label

    def saveModel(self, filename):
        self.model.save(filename)

    def plotLoss(self):
        plotLoss(self)

    def plotAccuracy(self):
        plotAccuracy(self)

    def plotWeight(self, layer_name, img_path):
        plotWeight(self, layer_name, img_path)


cal256_path = "./data/256_ObjectCategories"
urban_tribe_path = "./data/pictures_all"


print("\n-------------------------------------------------------------\n\n")
print("Training california 256 models\n\n")

cal256_models = []
for i in [16, 8, 4, 2]:
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    start_time = time.time()
    params = {"intermed_layer": "fc2",
              "valid_split": 0.1,
              "nb_per_class": i,
              "batch_size": 2,
              "nb_epoch": 0,
              "optimizer": "SGD",
              "lr": 1e-4,
              "momentum": 0.9,
              "nesterov": True,
              "early_stop": 0,
              "T": 1,}
    print("Parameters: ")
    for param in sorted(params.keys()):
        print(param, "\t", params[param])
    print("")

    model = VGG16TransferModel(cal256_path, **params)
    model.printModelArchitecture()
    cal256_models.append(model)

    elapsed_time = time.time() - start_time

    print("Training duration: ", str(elapsed_time), "s\n")
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


for i, model in enumerate(cal256_models):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    print("\nNumber of examples per class: ", str(2 ** (5 - i)), "\n")

    model.plotLoss()
    model.plotAccuracy()


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nclassification accuracy vs. number of samples used per class\n")
val_accs = [model.history.val_acc[-1] for model in cal256_models]
nper = [32, 16, 8, 4, 2]
plt.figure()
plt.semilogx(nper, val_accs, basex=2)
plt.figure()
plt.plot(nper, val_accs)


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nVisualize filters from first Convolution Layers of the trained model\n")
cal256_models[0].plotWeight("block1_conv1", "./data/256_ObjectCategories/001.ak47/001_0001.jpg")


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nVisualize filters from last Convolution Layers of the trained model\n")
cal256_models[0].plotWeight("block5_conv3", "./data/256_ObjectCategories/001.ak47/001_0001.jpg")


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nExperiment using the intermediate Convolutional Layers as input to the Softmax Layer\n")
params = {"intermed_layer": "block5_conv3",
          "valid_split": 0.05,
          "nb_per_class": 32,
          "batch_size": 16,
          "nb_epoch": 5,
          "optimizer": "SGD",
          "lr": 1e-4,
          "nesterov": True,
          "momentum": 0.9,
          "early_stop": 0,
          "T": 0}
cal256_inlayer_model = VGG16TransferModel(cal256_path, **params)


cal256_inlayer_model.plotLoss()
cal256_inlayer_model.plotAccuracy()
plt.show()


print("\n-------------------------------------------------------------\n\n")
print("Training california 256 models\n\n")

urban_tribe_models = []
for i in [32, 16, 8, 4, 2]:
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    start_time = time.time()
    params = {"intermed_layer": "fc2",
              "valid_split": 0.1,
              "nb_per_class": i,
              "batch_size": i,
              "nb_epoch": 30,
              "optimizer": "SGD",
              "lr": 1e-4,
              "momentum": 0.9,
              "nesterov": True,
              "early_stop": 3,
              "T": 0,}
    print("Parameters: ")
    for param in sorted(params.keys()):
        print(param, "\t", params[param])
    print("")

    model = VGG16TransferModel(urban_tribe_path, **params)
    urban_tribe_models.append(model)

    elapsed_time = time.time() - start_time

    print("Training duration: ", str(elapsed_time), "s\n")
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


for i, model in enumerate(urban_tribe_models):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    print("\nNumber of examples per class: ", str(2 ** (5 - i)), "\n")

    model.plotLoss()
    model.plotAccuracy()


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nclassification accuracy vs. number of samples used per class\n")
val_accs = [model.history.val_acc[-1] for model in urban_tribe_models]
nper = [32, 16, 8, 4, 2]
plt.figure()
plt.semilogx(nper, val_accs, basex=2)
plt.figure()
plt.plot(nper, val_accs)


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nVisualize filters from first Convolution Layers of the trained model\n")
urban_tribe_models[0].plotWeight("block1_conv1", "./data/256_ObjectCategories/001.ak47/001_0001.jpg")


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nVisualize filters from last Convolution Layers of the trained model\n")
urban_tribe_models[0].plotWeight("block5_conv3", "./data/256_ObjectCategories/001.ak47/001_0001.jpg")


print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
print("\nExperiment using the intermediate Convolutional Layers as input to the Softmax Layer\n")
params = {"intermed_layer": "block5_conv2",
          "valid_split": 0.1,
          "nb_per_class": 80,
          "batch_size": 16,
          "nb_epoch": 0,
          "optimizer": "SGD",
          "lr": 1e-4,
          "nesterov": True,
          "momentum": 0.9,
          "early_stop": 0,
          "T": 0}

ut_inlayer_model = VGG16TransferModel(urban_tribe_path, **params)
ut_inlayer_model.printModelArchitecture()


ut_inlayer_model.plotLoss()
ut_inlayer_model.plotAccuracy()


temperature_models = []
temps = [8]
val_accs = []
for T in temps:
    params = {"intermed_layer": "",
              "valid_split": 0.02,
              "nb_per_class": 16,
              "batch_size": 16,
              "nb_epoch": 8,
              "optimizer": "RMSprop",
              "lr": 1e0,
              "nesterov": True,
              "momentum": 0.9,
              "early_stop": 0,
              "T": T}

    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    print("\nTraining T =", str(T), "\n")

    model = VGG16TransferModel(cal256_path, **params)
    model.printModelArchitecture()
    temperature_models.append(model)
    val_accs.append(model.history.val_acc)

print(val_accs)

for i, model in enumerate(temperature_models):
    model.plotLoss()
    model.plotAccuracy()

val_accs1 = [val_acc[-1] for val_acc in val_accs]
plt.figure()
plt.semilogx(temps, val_accs1, basex=2)
plt.figure()
plt.plot(temps, val_accs1)

val_accs2 = [max(val_acc) for val_acc in val_accs]
plt.figure()
plt.semilogx(temps, val_accs2, basex=2)
plt.figure()
plt.plot(temps, val_accs2)
plt.show()
