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


def load_config(input_path, prefix, length):
    CP = []
    for i in xrange(length):
        cp = ConfigParser.ConfigParser()
        cp.read(prefix + '_' + str(i) + '.ini')
        CP.append(cp)
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    CP.append(cp)
    return CP


def log(s, label='INFO'):
    sys.stderr.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

def load_data(cp, length, train):
    log('Loading data........')
    if cp[length].get('Experiment', 'inputfile') == '':
        [X1, labels1] = utils.load_mnist(
            dataset='training', path='/home/ubuntu/data/mnist/')
        [X2, labels2] = utils.load_mnist(
            dataset='testing', path='/home/ubuntu/data/mnist/')
        X = np.concatenate((X1, X2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p].astype(np.float32) * 0.02
            labels = labels[p]
            prefix = cp[length].get('Experiment','prefix')
            np.save(prefix + '_sda_random_perm.npy', p)
        return [X, labels]
    else:
        X = np.load(cp[length].get('Experiment', 'inputfile'))
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p]
            prefix = cp[length].get('Experiment', 'prefix')
            np.save(prefix + '_sda_random_perm.npy', p)
        return X
    log('DONE........')


def init(cp, dataset, length):
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

    print '> Deep neural net Topology'
    print '----------------------------------'
    print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network))
    print '----------------------------------'

    # Create train function
    learning_rate = T.scalar(name='learning_rate')
    index = T.lscalar()
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.squared_error(
        prediction, input_layer.input_var).mean()
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.momentum(
        cost, params, learning_rate=learning_rate)
    train = theano.function(
        inputs=[index, learning_rate], outputs=cost, updates=updates, givens={input_layer.input_var: input_var[index:index + batch_size, :]})

    deep_epochs = int(cp[length].get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp[length].get('NeuralNetwork', 'learningrate'))
    # train
    print '> Deep neural net trainining'
    for epoch in xrange(deep_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            loss = train(row, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0] / batch_size)
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss), label='DNT')
        if (epoch % lr_decay == 0 and epoch != 0):
            base_lr = base_lr / 10
        if (epoch % 100 == 0) and (epoch != 0):
           np.save(prefix + '_sda_model.npy',
            lasagne.layers.get_all_param_values(network))
    input_layer.input_var = input_var
    np.save(prefix + '_sda_model.npy',
            lasagne.layers.get_all_param_values(network))
    np.save(prefix + '_sda_W1.npy', encoder_layer.W.eval())
    np.save(prefix + '_sda_W2.npy', network.W.eval())
    np.save(prefix + '_sda_b1.npy', encoder_layer.b.eval())
    np.save(prefix + '_sda_b2.npy', network.b.eval())
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    log(hidden.shape)
    np.save(prefix + '_sda_hidden.npy', hidden)
    np.save(prefix + '_sda_output.npy',
            lasagne.layers.get_output(network).eval())


def init_pretrained(cp, dataset, length):
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
    lasagne.layers.set_all_param_values(
        network, np.load(prefix + '_sda_model.npy'))
    input_layer.input_var = input_var
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    np.save(prefix + '_sda_pretrained_hidden.npy', hidden)
    output = lasagne.layers.get_output(network).eval()
    np.save(prefix + '_sda_pretrained_output.npy', output)


def main(path, length, train, pref):
    cp = load_config(path, pref, length)
    try:
        [X, labels] = load_data(cp, length, train)
    except:
        X = load_data(cp, length, train)
    if train == 'train':
        init(cp, X, length)
    else:
        init_pretrained(cp, X, length)


from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    parser.add_argument('-n', '--number', required=True, type=int,
                        help='number of autoencoders ')
    parser.add_argument('-t', '--train', required=True, type=str,
                        help='training/testing')
    parser.add_argument('-p', '--prefix', required=True, type=str,
                        help='sbufiles prefix')
    opts = parser.parse_args()
    getter = attrgetter('input', 'number', 'train','prefix')
    inp, length, train, pref= getter(opts)
    main(inp, length, train, pref)
