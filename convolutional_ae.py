import os
os.environ[
    'THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
# os.environ['THEANO_FLAGS'] = 'device=cpu'
import sys
import ConfigParser
import numpy as np
import theano
from theano import tensor as T
import datetime
from datetime import datetime
import dataset_utils as utils
import lasagne
from Unpool2DLayer import Unpool2DLayer
import dataset_utils as utils
from sklearn.utils.linear_assignment_ import linear_assignment
from lasagne.regularization import regularize_layer_params, l2
from modeltemplate import Model


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def load_config(input_path):
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    return cp


def log(s, label='INFO'):
    sys.stderr.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()


def load_data(cp, train):
    log('Loading data........')
    if cp.get('Experiment', 'inputfile') == '':
        [X1, labels1] = utils.load_mnist(
            dataset='training', path='/home/ubuntu/data/mnist/')
        [X2, labels2] = utils.load_mnist(
            dataset='testing', path='/home/ubuntu/data/mnist/')
        X = np.concatenate((X1, X2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        # if train == 'train':
        X = X.astype(np.float32) * 0.02
        #     np.random.shuffle(X)
        return [X[0:10000], labels]
    else:
        X = np.load(cp.get('Experiment', 'inputfile'))
        # if train == 'train':
        #     p = np.random.permutation(X.shape[0])
        #     X = X[p]
        #     prefix = cp.get('Experiment', 'prefix')
        #     num = cp.get('Experiment', 'num')
        #     np.save(prefix + '_' + num + 'random_perm.npy', p)
        return X
    log('DONE........')


def init(cp, dataset):
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    log(dataset.shape)
    print dataset.shape
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    conv_filters = int(cp.get('NeuralNetwork', 'convfilters'))
    deconv_filters = conv_filters
    filter_sizes = int(cp.get('NeuralNetwork', 'filtersize'))
    featurex = int(cp.get('NeuralNetwork', 'feature_x'))
    featurey = int(cp.get('NeuralNetwork', 'feature_y'))
    channels = int(cp.get('NeuralNetwork', 'channels'))
    pool_size = int(cp.get('NeuralNetwork', 'pool'))
    input_layer = network = lasagne.layers.InputLayer(shape=(None, featurex * featurey),
                                                      input_var=input_var)
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], channels, featurex, featurey))
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=conv_filters, filter_size=(
                                             filter_sizes, filter_sizes),
                                         stride=int(cp.get('NeuralNetwork', 'stride')))
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.Conv2DLayer(incoming=network,
                                             num_filters=conv_filters, filter_size=(
                                                 dual_conv, dual_conv),
                                             stride=int(cp.get('NeuralNetwork', 'dualstride')))
    except:
        pass
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network, pool_size=(pool_size, pool_size))
    pool_shape = lasagne.layers.get_output_shape(network)
    flatten = network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec2')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=lasagne.layers.get_output_shape(flatten)[
                                            1],
                                        )
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], deconv_filters, pool_shape[2], pool_shape[3]))
    network = lasagne.layers.Upscale2DLayer(incoming=network, scale_factor=(pool_size, pool_size))
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                            num_filters=1, filter_size=(dual_conv, dual_conv), stride=int(cp.get('NeuralNetwork', 'dualstride')), nonlinearity=None)
    except:
        pass
    network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                         num_filters=1, filter_size=(filter_sizes, filter_sizes), stride=int(cp.get('NeuralNetwork', 'stride')), nonlinearity=None)
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network))

    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.squared_error(
        prediction, input_layer.input_var).mean()
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate, momentum=0.975)
    train = theano.function(
        inputs=[index, learning_rate], outputs=cost,
        updates=updates, givens={input_layer.input_var: input_var[index:index + batch_size, :]})
    prefix = cp.get('Experiment', 'prefix')
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    # Start training
    lr_decay = int(cp.get('NeuralNetwork', 'lrepochdecay'))
    for epoch in xrange(lw_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            loss = train(row, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0] * 1.0 / batch_size)
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss),
                label='CONV')
        if (epoch % lr_decay == 0 and epoch != 0):
            base_lr = base_lr / 10
        if (epoch % 100 == 0) and (epoch != 0):
            utils.save(prefix + '_conv.zip', network)
    input_layer.input_var = input_var
    np.save(prefix + '_model.npy',
            lasagne.layers.get_all_param_values(network))
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    log(hidden.shape)
    np.save(prefix + '_hidden.npy', hidden)


def init_pretrained(cp, dataset):
    print dataset.shape
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    conv_filters = int(cp.get('NeuralNetwork', 'convfilters'))
    deconv_filters = conv_filters
    filter_sizes = int(cp.get('NeuralNetwork', 'filtersize'))
    featurex = int(cp.get('NeuralNetwork', 'feature_x'))
    featurey = int(cp.get('NeuralNetwork', 'feature_y'))
    channels = int(cp.get('NeuralNetwork', 'channels'))
    pool_size = int(cp.get('NeuralNetwork', 'pool'))
    input_layer = network = lasagne.layers.InputLayer(shape=(None, featurex * featurey),
                                                      input_var=input_var)
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], channels, featurex, featurey))
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=conv_filters, filter_size=(
                                             filter_sizes, filter_sizes),
                                         stride=int(cp.get('NeuralNetwork', 'stride')))
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.Conv2DLayer(incoming=network,
                                             num_filters=conv_filters, filter_size=(
                                                 dual_conv, dual_conv),
                                             stride=int(cp.get('NeuralNetwork', 'dualstride')))
    except:
        pass
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network, pool_size=(pool_size, pool_size))
    pool_shape = lasagne.layers.get_output_shape(network)
    flatten = network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec2')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=lasagne.layers.get_output_shape(flatten)[
                                            1],
                                        )
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], deconv_filters, pool_shape[2], pool_shape[3]))
    network = lasagne.layers.Upscale2DLayer(incoming=network, scale_factor=(pool_size, pool_size))
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                            num_filters=1, filter_size=(dual_conv, dual_conv), stride=int(cp.get('NeuralNetwork', 'dualstride')), nonlinearity=None)
    except:
        pass
    network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                         num_filters=1, filter_size=(filter_sizes, filter_sizes), stride=int(cp.get('NeuralNetwork', 'stride')), nonlinearity=None)
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    model = Model(input_layer=input_layer, encoder_layer=encoder_layer,
                  decoder_layer=network, network=network)
    model.save('GHT_700_shallow.zip')
    lasagne.layers.set_all_param_values(
        network, np.load(prefix + '_model.npy'))
    input_layer.input_var = input_var
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    np.save(prefix + '_pretrained_hidden.npy', hidden)
    output = lasagne.layers.get_output(network).eval()
    np.save(prefix + '_pretrained_output.npy', output)


def main(path, train):
    cp = load_config(path)
    try:
        [X, labels] = load_data(cp, train)
    except:
        X = load_data(cp, train)
    if train == 'train':
        init(cp, X)
    else:
        init_pretrained(cp, X)


from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    parser.add_argument('-t', '--train', required=True, type=str,
                        help='training/testing')
    opts = parser.parse_args()
    getter = attrgetter('input', 'train')
    inp, train = getter(opts)
    main(inp, train)
