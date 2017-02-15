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
import sdc.image_processing.feature_extraction as fe
import sdc.image_processing.feature_extraction_settings as fes
from keras.layers.normalization import BatchNormalization

import tensorflow as tf
import _pickle as pkl
from sklearn.preprocessing import StandardScaler


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


class NNClassifier:
    def __init__(self, settings=None):

        hist_bins = 32
        spatial_size = (32, 32)
        pix_per_cell = 8
        orient = 9
        cell_per_block = 2

        self.classifier = None
        self.settings = settings
        self.scaler = None
        if self.settings is None:
            settings = fes.FeatureExtractionSettings()
            settings.color_space = 'YCrCb'
            settings.pix_per_cell = 8
            settings.spatial_size = (32, 32)
            settings.cell_per_block = 2
            settings.spatial_features = True
            settings.hist_features = True
            settings.equalize_histogram = False
            settings.hist_bins = 32
            settings.orient = 9
            self.settings = settings
        else:
            self.settings = settings

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
        self.model = model
        return model

    def _model(self):
        img_width, img_height = 64, 64
        model = Sequential()

        model.add(Dense(2048, input_dim=8460, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='elu'))
        model.add(Dropout(0.8))
        model.add(Dense(1, activation='sigmoid'))
        self.model = model

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def save(self):
        model_json = self.model.to_json()
        with open("./model.json", "w") as json_file:
            json.dump(model_json, json_file)
        self.model.save_weights("./model.h5")
        with open('./scaler.pkl', 'wb') as f:
            pkl.dump(self.scaler, f)
        print("Saved model to disk")

    def load(self):
        with open('./scaler.pkl', 'rb') as f:
            self.scaler = pkl.load(f)
        with open('./model.json', 'r') as jfile:
            self.model = model_from_json(json.load(jfile))

        self.compile()
        self.model.load_weights('./model.h5')
        self.X_scaler = self.scaler
        self.classifier = self.model



    def predict(self, image):
        """ Predict if given image contains object of interest """
        # Extract features
        features = fe.single_img_features(image, self.settings)
        scaled_X = self.scaler.transform(np.array(features).reshape(1, -1))
        #features = features[None,:]
        # Apply scaler
        #scaled_X = self.X_scaler.transform(np.array(features).reshape(1, -1))
        # Predict using classifier
        return self.model.predict(scaled_X, batch_size=1)



    def train(self, file_list, labels, test_size=0.2, nb_epoch=30, batch_size=128):
        with open('dataset.pkl', 'rb') as f:
            X_all = pkl.load(f)
            Y_all = pkl.load(f)
        print('X_all.shape {}'.format(X_all.shape))
        self.scaler = StandardScaler().fit(X_all)
        X_all = self.scaler.transform(X_all)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_all, Y_all, test_size=test_size, random_state=100)
        self._model()
        self.compile()
        self.model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size)
        score = self.model.evaluate(X_test, Y_test)
        print('Score: {}'.format(score))


def get_list():
    vehicles = np.array(glob.glob('training_data/vehicles/*/*'))
    y_vehicles = np.zeros(vehicles.shape) + 1
    non_vehicles = np.array(glob.glob('training_data/non-vehicles/*/*'))
    y_non_vehicles = np.zeros(non_vehicles.shape)
    #X_data = np.concatenate((vehicles, non_vehicles))
    #Y_data = np.concatenate((y_vehicles, y_non_vehicles))
    return vehicles, y_vehicles, non_vehicles, y_non_vehicles

def prepare_images(settings):
    v_x, v_y, n_x, n_y = get_list()
    labels = np.concatenate((v_y, n_y))
    car_features = fe.extract_features(v_x, settings)
    notcar_features = fe.extract_features(n_x, settings)
    X_all = np.concatenate((car_features, notcar_features))
    with open('dataset.pkl', 'wb') as f:
        pkl.dump(X_all, f)
        pkl.dump(labels, f)


def build_images(x):
    images = np.zeros((len(x), 64, 64, 3))
    for idx, img_fname in enumerate(x):
        im = cv2.imread(img_fname)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        images[idx] = im
    return images

def do_all(nb_epoch=30, batch_size=256):
    clf = NNClassifier()
    x, y = get_list()
    clf.train(x, y, nb_epoch=nb_epoch, batch_size=batch_size)
    clf.save()

