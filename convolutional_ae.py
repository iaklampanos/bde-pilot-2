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


def load_data(cp, train):
    log('Loading data........')
    if cp.get('Experiment', 'inputfile') == '':
        [X1, labels1] = utils.load_mnist(
            dataset='training', path='/mnt/disk1/thanasis/autoencoder/')
        [X2, labels2] = utils.load_mnist(
            dataset='testing', path='/mnt/disk1/thanasis/autoencoder/')
        X = np.concatenate((X1, X2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        # if train == 'train':
        X = X.astype(np.float32) * 0.02
        #     np.random.shuffle(X)
        return [X[0:1000], labels]
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
    batch_size = 100
    log(dataset.shape)
    print dataset.shape
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    conv_filters = 30
    deconv_filters = 30
    filter_sizes = 3
    # batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    # lr_decay = int(cp.get('NeuralNetwork', 'lrepochdecay'))
    input_layer = network = lasagne.layers.InputLayer(shape=(None, 784),
                                                      input_var=input_var)
    network = lasagne.layers.ReshapeLayer(
            incoming=network, shape=([0], 1, 28 , 28))
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=conv_filters, filter_size=(
                                             filter_sizes, filter_sizes),
                                         stride=1, pad='same',nonlinearity=None)
    network = lasagne.layers.MaxPool2DLayer(incoming=network, pool_size=(2, 2))
    network = Unpool2DLayer(incoming=network, ds=(2, 2))
    flatten = network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    encoder_layer = network = lasagne.layers.DenseLayer(incoming=network,
                                                        num_units=40,
                                                        )
    hidden = network = lasagne.layers.DenseLayer(incoming=network,
                                                 num_units=deconv_filters * (28 + filter_sizes - 1) ** 2 / 4,
                                                 )
    network = lasagne.layers.ReshapeLayer(
            incoming=network, shape=([0], deconv_filters, (28 + filter_sizes - 1) / 2, (28 + filter_sizes - 1) / 2 ))
    network = Unpool2DLayer(incoming=network,ds=(2,2))
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=1, filter_size=(filter_sizes, filter_sizes),nonlinearity=None)
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network))

    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.squared_error(
        prediction, input_layer.input_var).mean()
    # l2_penalty = regularize_layer_params(network, l2) * (0.001) / 2
    # cost = cost - l2_penalty
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.momentum(
        cost, params, learning_rate=learning_rate, momentum=0.975)
    train = theano.function(
        inputs=[index, learning_rate], outputs=cost,
        updates=updates, givens={input_layer.input_var: input_var[index:index + batch_size, :]})

    # lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    # base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    base_lr = 0.1
    lw_epochs = 20
    lr_decay = 73
    # Start training
    # num = cp.get('Experiment', 'num')
    num = 0
    for epoch in xrange(lw_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            loss = train(row, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0]*1.0 / batch_size)
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss),
                label='LWT-Layer' + str(num))
        if (epoch % lr_decay == 0 and epoch != 0):
            base_lr = base_lr / 10
        if (epoch % 100 == 0) and (epoch != 0):
            utils.save('autoenc_' + str(num) + '.zip', network)
    input_layer.input_var = input_var
    a_out = lasagne.layers.get_output(network).eval()
    dataset = dataset.reshape(100, 784)
    a_out = a_out.reshape(100, 784)
    for i in range(0, 100):
        utils.plot_pixel_image(dataset[i, :], a_out[i, :], 28, 28)
    [X1, labels1] = utils.load_mnist(
        dataset='training', path='/mnt/disk1/thanasis/autoencoder/')
    [X2, labels2] = utils.load_mnist(
        dataset='testing', path='/mnt/disk1/thanasis/autoencoder/')
    X = np.concatenate((X1, X2), axis=0)
    labels = np.concatenate((labels1, labels2), axis=0)
    labels = labels.reshape(70000)
    kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=-1)
    hidden = np.load('MNIST_0_pretrained_hidden.npy')
    cluster_prediction = kmeans.fit_predict(hidden)
    acc = cluster_acc(labels, cluster_prediction)
    print 'LW', acc
    input_layer.input_var = input_var
    kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=-1)
    hidden = lasagne.layers.get_output(encoder_layer)
    cluster_prediction = kmeans.fit_predict(hidden)
    acc = cluster_acc(labels, cluster_prediction)
    print 'Conv', acc
    # prefix = cp.get('Experiment', 'prefix')
    # np.save(prefix + '_' + num + '_model.npy',
    #         lasagne.layers.get_all_param_values(network))
    # np.save(prefix + '_' + num + '_W1.npy', encoder_layer.W.eval())
    # np.save(prefix + '_' + num + '_W2.npy', network.W.eval())
    # np.save(prefix + '_' + num + '_b1.npy', encoder_layer.b.eval())
    # np.save(prefix + '_' + num + '_b2.npy', network.b.eval())
    # hidden = lasagne.layers.get_output(encoder_layer).eval()
    # log(hidden.shape)
    # np.save(prefix + '_hidden.npy', hidden)


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