from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, ELU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.preprocessing import normalize
import cv2
import numpy as np
import glob
import json
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [shape[:1] // parts, shape[1:]])
        stride = tf.concat(0, [shape[:1] // parts, shape[1:] * 0])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


class CNNClassifier:
    def __init__(self):
        self.classifier = None

    def get_model(self, parallel=False):
        model = Sequential()
        #model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(64, 64, 3)))
        model.add(Convolution2D(8, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv1'))
        model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv2'))
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv3'))
        model.add(Flatten())
        model.add(ELU())
        model.add(Dense(1024, activation='elu'))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(512, activation='elu'))
        model.add(Dropout(.5))
        model.add(Dense(1, name='output'))
        model.add(Activation('sigmoid'))
        if parallel:
            model = make_parallel(model, 2)
        #model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def _model(self):
        img_width, img_height = 64, 64
        model = Sequential()
        model.add(Convolution2D(8, 3, 3, input_shape=(img_width, img_height, 3)))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #model.add(Convolution2D(16, 3, 3))
        #model.add(Activation('elu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        #model.add(Convolution2D(32, 3, 3))
        #model.add(Activation('elu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        #model = make_parallel(model, 2)
        self.model = model

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', class_mode='binary',
                      metrics=['accuracy'])

    def save(self):
        model_json = self.model.to_json()
        with open("./model.json", "w") as json_file:
            json.dump(model_json, json_file)
        self.model.save_weights("./model.h5")
        print("Saved model to disk")

    def load(self):
        with open('./model.json', 'r') as jfile:
            self.model = model_from_json(json.load(jfile))

        self.compile()
        self.model.load_weights('./model.h5')

    def get_list(self):
        vehicles = np.array(glob.glob('training_data/vehicles/*/*'))
        y_vehicles = np.zeros(vehicles.shape) + 1
        non_vehicles = np.array(glob.glob('training_data/non-vehicles/*/*'))
        y_non_vehicles = np.zeros(non_vehicles.shape)
        X_data = np.concatenate((vehicles, non_vehicles))
        Y_data = np.concatenate((y_vehicles, y_non_vehicles))
        return X_data, Y_data

    def predict(self, image):
        #img = np.copy(image)
        #img = cv2.resize(img, (64, 64))
        x = image[None, :, :, :]
        result = self.model.predict(x, 1)
        return result

    def train(self, file_list, labels, test_size=0.2, nb_epoch=30, batch_size=128):
        X_train, X_test, Y_train, Y_test = train_test_split(file_list, labels, test_size=test_size, random_state=100)

        test_images = build_images(X_test)
        train_images = build_images(X_train)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.05,
            zoom_range=0.05,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=5,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow(train_images, Y_train, batch_size)
        test_generator = test_datagen.flow(test_images, Y_test, batch_size)

        nb_train_samples = (batch_size-1)*100
        nb_validation_samples = (batch_size-1)*20

        #self.get_model(parallel=False)
        self._model()
        self.compile()

        self.model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch, show_accuracy=True,
            validation_data=test_generator,
            nb_val_samples=nb_validation_samples)

def build_images(x):
    images = np.zeros((len(x), 64, 64, 3))
    for idx, img_fname in enumerate(x):
        im = cv2.imread(img_fname)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        images[idx] = im
    return images

def do_all(nb_epoch=30, batch_size=256):
    clf = CNNClassifier()
    x, y = clf.get_list()
    clf.train(x, y, nb_epoch=nb_epoch, batch_size=batch_size)
    clf.save()

