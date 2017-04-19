from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano as th
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import squared_error
from lasagne.regularization import l2
from netcdf_subset import netCDF_subset
from Dataset_transformations import Dataset_transformations
import dataset_utils as utils
from shape import ReshapeLayer, Unpool2DLayer
from lasagne.objectives import squared_error
from lasagne.nonlinearities import tanh
import pickle
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
import os
import urllib
import gzip
import cPickle
from IPython.display import Image as IPImage
from PIL import Image
import dataset_utils as utils


class ConvAutoencoder(object):

    def __init__(self, X_train, conv_filters, deconv_filters, filter_sizes, epochs,
                 hidden_size, channels, stride, corruption_level, l2_level,
                 samples, features_x, features_y):
        self._dataset = X_train
        self.input_var = X_train
        self.conv_filters = conv_filters
        self.deconv_filters = deconv_filters
        self.filter_sizes = filter_sizes
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.m = samples
        self.ae = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('conv', layers.Conv2DLayer),
                ('pool', layers.MaxPool2DLayer),  # max pool 2D/3D/4D ?
                ('flatten', ReshapeLayer),  # output_dense
                ('encode_layer', layers.DenseLayer),
                ('hidden', layers.DenseLayer),  # output_dense
                ('unflatten', ReshapeLayer),
                ('unpool', Unpool2DLayer),
                ('deconv', layers.Conv2DLayer),
                ('output_layer', ReshapeLayer),
            ],
            input_shape=(None, channels, features_x, features_y),
            input_var=self.get_corrupted_input(
                self.input_var, corruption_level),
            conv_num_filters=self.conv_filters, conv_filter_size=(
                filter_sizes, filter_sizes),
            conv_stride=stride,
            # conv_border_mode="valid", removed from latest version
            conv_nonlinearity=None,
            pool_pool_size=(2, 2),
            flatten_shape=(([0], -1)),  # not sure if necessary?
            encode_layer_num_units=self.hidden_size,
            hidden_num_units=self.deconv_filters * \
            (features_x + filter_sizes - 1) ** 2 / 4,
            unflatten_shape=(
                ([0], deconv_filters, (features_x + filter_sizes - 1) / 2, (features_y + filter_sizes - 1) / 2)),
            unpool_ds=(2, 2),
            deconv_num_filters=channels, deconv_filter_size=(
                filter_sizes, filter_sizes),
            # deconv_border_mode="valid",
            deconv_nonlinearity=None,
            output_layer_shape=(([0], -1)),
            update_learning_rate=0.01,
            update_momentum=0.975,
            objective_l2=l2_level,
            objective_loss_function=squared_error,
            batch_iterator_train=BatchIterator(batch_size=500),
            regression=True,
            max_epochs=epochs,
            verbose=1,
        )

    def train(self):
        X_out = self._dataset.reshape((self._dataset.shape[0], -1))
        self.ae.fit(self._dataset, X_out)
        self.decoded = self.test(self._dataset)
        self.hidden = self.get_hidden(self._dataset)

    def test(self, X_pred):
        X_pred_shape = X_pred.shape
        return self.ae.predict(X_pred).reshape(X_pred_shape)

    def get_corrupted_input(self, input, corruption_level):
        return RandomStreams(np.random.RandomState().randint(2 ** 30)).binomial(size=input.shape, n=1,
                                                                                p=1 - corruption_level,
                                                                                dtype=th.config.floatX) * input

    def get_output_from_nn(self, last_layer, X):
        indices = np.arange(164, X.shape[0], 164)
        sys.stdout.flush()

        # not splitting into batches can cause a memory error
        X_batches = np.split(X, indices)
        out = []
        for count, X_batch in enumerate(X_batches):
            out.append(layers.helper.get_output(last_layer, X_batch).eval())
            sys.stdout.flush()
        return np.vstack(out)

    def get_hidden(self, X, grid_x=self._dataset.shape[4], grid_y=self._dataset.shape[5]):
        try:
            encode_layer_index = map(
                lambda pair: pair[0], self.ae.layers).index('encode_layer')
            encode_layer = self.ae.get_all_layers()[encode_layer_index]
        except:
            X = X.reshape(X.shape[1], 1, grid_x, grid_y)
            encode_layer_index = map(
                lambda pair: pair[0], self.ae.layers).index('encode_layer')
            encode_layer = self.ae.get_all_layers()[encode_layer_index]
        return self.get_output_from_nn(encode_layer, X)

    def get_output(self, X):
        return self.get_output_from_nn(output_layer, X)

    def save(self, filename='ConvAutoencoder.zip'):
        utils.save(filename, self)

    def load(self, filename='ConvAutoencoder.zip'):
        self = utils.load(filename)
