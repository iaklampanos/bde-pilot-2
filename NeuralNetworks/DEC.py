import os
os.environ[
    'THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'

import sys
import ConfigParser
import numpy as np
import theano
from theano import tensor as T
import datetime
from datetime import datetime
import dataset_utils as utils
import lasagne
from sklearn.cluster import KMeans
from ClusteringLayer import ClusteringLayer


def load_config(input_path, length):
    CP = []
    for i in xrange(length):
        cp = ConfigParser.ConfigParser()
        cp.read('autoenc' + str(i) + '.ini')
        CP.append(cp)
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    CP.append(cp)
    return CP

from sklearn.utils.linear_assignment_ import linear_assignment


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def log(s, label='INFO'):
    sys.stderr.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')


def load_data(cp):
    log('Loading data........')
    if cp[length].get('Experiment', 'inputfile') == '':
        [X1, labels1] = utils.load_mnist(
            dataset='training', path='/mnt/disk1/thanasis/autoencoder/')
        [X2, labels2] = utils.load_mnist(
            dataset='testing', path='/mnt/disk1/thanasis/autoencoder/')
        X = np.concatenate((X1, X2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        return [X, labels]
    else:
        X = np.load(cp[length].get('Experiment', 'inputfile'))
        return X
    log('DONE........')


def target_dist(q):
    # q = (q.T / q.sum(axis=1)).T
    p = (q**2)
    p = (p.T / p.sum(axis=1)).T
    return p


def init(cp, dataset, length, clusters, n_init=20, labels=None):
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    batch_size = int(cp[length].get('NeuralNetwork', 'batchsize'))
    lr_decay = int(cp[length].get('NeuralNetwork', 'lrepochdecay'))
    prefix = cp[length].get('Experiment', 'prefix')
    log(dataset.shape)
    # Create deep network
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    input_layer = network = lasagne.layers.InputLayer(shape=(None, int(cp[0].get('NeuralNetwork', 'inputlayer'))),
                                                      input_var=input_var)
    # Deep encoders
    for i in xrange(length):
        enc_act = cp[i].get('NeuralNetwork', 'encoderactivation')
        network = lasagne.layers.DenseLayer(incoming=input_layer if i == 0 else network,
                                            num_units=int(
                                                cp[i].get('NeuralNetwork', 'hiddenlayer')),
                                            W=np.load(
                                                prefix + '_' + str(i) + '_W1.npy'),
                                            b=np.load(
                                                prefix + '_' + str(i) + '_b1.npy'),
                                            nonlinearity=relu if enc_act == 'ReLU' else linear)
    encoder_layer = network

    # Deep decoders
    for i in reversed(xrange(length)):
        dec_act = cp[i].get('NeuralNetwork', 'decoderactivation')
        network = lasagne.layers.DenseLayer(incoming=network,
                                            num_units=int(
                                                cp[i].get('NeuralNetwork', 'outputlayer')),
                                            W=np.load(
                                                prefix + '_' + str(i) + '_W2.npy'),
                                            b=np.load(
                                                prefix + '_' + str(i) + '_b2.npy'),
                                            nonlinearity=relu if dec_act == 'ReLU' else linear)
    # lasagne.layers.set_all_param_values(
    #     network, np.load(prefix + '_sda_model.npy'))
    dec = encoder_layer
    kmeans = KMeans(n_clusters=clusters, n_init=n_init, n_jobs=-1)
    input_layer.input_var = input_var
    hidden = lasagne.layers.get_output(dec).eval()
    cluster_prediction = kmeans.fit_predict(hidden)
    print cluster_prediction
    if not(labels is None):
        print cluster_acc(labels, cluster_prediction)
    centroids = kmeans.cluster_centers_
    dec = ClusteringLayer(incoming=dec, num_units=clusters,
                          W=centroids,
                          shape=(clusters, hidden.shape[1]))
    base_lr = 0.01
    x = T.matrix('x')
    prediction = lasagne.layers.get_output(dec)
    x = target_dist(prediction)
    cost = lasagne.objectives.categorical_crossentropy(prediction, x).mean()
    params = lasagne.layers.get_all_params(dec, trainable=True)
    updates = lasagne.updates.momentum(cost, params, learning_rate=base_lr)
    train_fn = theano.function(
        inputs=[], outputs=[cost, prediction.argmax(1), prediction],
        updates=updates, givens={input_layer.input_var: input_var})
    print '> Cluster finetuning'
    train = True
    epochs = 0
    while train:
        input_layer.input_var = input_var
        loss, new_pred, prediction = train_fn()
        # print new_pred.shape
        # print cluster_prediction.shape
        print new_pred
        if not(labels is None):
            print cluster_acc(labels, new_pred)
        delta = ((new_pred != cluster_prediction).sum().astype(
            np.float32) / new_pred.shape[0])
        # print new_pred
        if (epochs > 100000) or (delta < 0.01):
            train = False
            np.save(prefix + '_dec_model.npy',
                    lasagne.layers.get_all_param_values(dec))
            np.save(prefix + '_dec_centroids.npy', dec.W.eval())
        else:
            cluster_prediction = new_pred
        if epochs % 10 == 0:
            # print '> Epoch ' + str(epochs) + ' Change in label
            # assignment.... : ' + str(delta)
            log(str(epochs) + ' ' + str(loss) + ' ' + str(delta),
                label='DEC')
        if (epochs % 1000 == 0) and (epochs != 0):
            np.save(prefix + '_dec_model.npy',
                    lasagne.layers.get_all_param_values(dec))
            np.save(prefix + '_dec_centroids.npy', dec.W.eval())
        epochs += 1


def main(path, length, clusters):
    cp = load_config(path, length)
    try:
        [X, labels] = load_data(cp)
    except:
        X = load_data(cp)
    init(cp, X, length, clusters,labels=labels)

from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    parser.add_argument('-n', '--number', required=True, type=int,
                        help='number of autoencoders ')
    parser.add_argument('-c', '--clusters', required=True, type=int,
                        help='number of clusters ')
    opts = parser.parse_args()
    getter = attrgetter('input', 'number', 'clusters')
    inp, length, clusters = getter(opts)
    main(inp, length, clusters)
