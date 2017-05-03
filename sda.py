import sys
import numpy as np
import theano as th
import theano.tensor as T
from lasagne.regularization import regularize_layer_params_weighted, l2
from lasagne.regularization import regularize_layer_params
from sklearn.utils.linear_assignment_ import linear_assignment
import lasagne
from ClusteringLayer import ClusteringLayer
from sklearn.cluster import KMeans
import gzip
import cPickle
import dataset_utils as utils
import datetime
from datetime import datetime


def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind])*1.0/y_pred.size


class sda(object):

    def __init__(self, feature_shape, dims, lr_epoch_decay,
                 learning_rate,  mini_batch_size=1000, corruption_factor=0.3):
        self._feature_shape = feature_shape
        self._dims = dims
        self.learning_rate = learning_rate
        self._corruption_factor = corruption_factor
        # self._encoders = []
        # self._decoders = []
        # Init activations
        self.enc_act = []
        for i in range(0, len(self._dims)):
            if i == len(self._dims) - 1:
                self.enc_act.append(lasagne.nonlinearities.linear)
            else:
                self.enc_act.append(lasagne.nonlinearities.rectify)
        self.dec_act = []
        for i in range(0, len(self._dims)):
            if i == 0:
                self.dec_act.append(lasagne.nonlinearities.linear)
            else:
                self.dec_act.append(lasagne.nonlinearities.rectify)
        self._layer_wise_autoencoders = []
        self.mini_batch_size = mini_batch_size
        self.lr_epoch_decay = lr_epoch_decay

    def init_lw(self, dataset, filename='layerwise_models.zip', lw_epochs=50000):
        # Init layer wise
        input_var = T.matrix('input_var')
        index = T.lscalar()
        current_input = th.shared(name='X', value=np.asarray(dataset,
                                                             dtype=th.config.floatX),
                                  borrow=True)
        for i in range(0, len(self._dims)):
            base_lr = self.learning_rate
            # Init lw autoencoders
            # Initialize layers
            input_layer = lasagne.layers.InputLayer(
                shape=(None, self._dims[i][0]), input_var=input_var)
            dropout_layer = lasagne.layers.DropoutLayer(incoming=input_layer,
                                                        p=self._corruption_factor)
            encoder_layer = lasagne.layers.DenseLayer(incoming=dropout_layer,
                                                      num_units=self._dims[
                                                          i][1],
                                                      W=lasagne.init.Normal(),
                                                      nonlinearity=self.enc_act[i])
            decoder_layer = lasagne.layers.DenseLayer(incoming=encoder_layer,
                                                      num_units=self._dims[
                                                          i][2],
                                                      W=lasagne.init.Normal(),
                                                      nonlinearity=self.dec_act[i])

            # Create train function
            learning_rate = T.scalar(name='learning_rate')
            prediction = lasagne.layers.get_output(decoder_layer)
            cost = lasagne.objectives.squared_error(
                prediction, input_var).mean()
            params = lasagne.layers.get_all_params(
                decoder_layer, trainable=True)
            updates = lasagne.updates.momentum(
                cost, params, learning_rate=learning_rate)
            train = th.function(
                inputs=[index, learning_rate], outputs=cost,
                updates=updates, givens={input_layer.input_var: current_input[index:index + self.mini_batch_size, :]})

            # Start training
            for epoch in xrange(lw_epochs):
                epoch_loss = 0
                for row in xrange(0, dataset.shape[0], self.mini_batch_size):
                    loss = train(row, base_lr)
                    epoch_loss += loss
                epoch_loss = float(epoch_loss) / (dataset.shape[0]/self.mini_batch_size)
                if epoch % 10 == 0:
                    log(str(epoch) + ' ' + str(epoch_loss),
                        label='LWT-Layer' + str(i))
                    # print '>'+str(datetime.datetime.utcnow())+ ' Epoch ' +
                    # str(epoch) + ' Loss function value.... : ' + str(loss)
                if epoch % self.lr_epoch_decay == 0:
                    base_lr = base_lr / 10
            self.save(filename)
            input_layer.input_var = current_input
            hidden = lasagne.layers.get_output(encoder_layer).eval()
            current_input = th.shared(name='X', value=np.asarray(hidden,
                                                                 dtype=th.config.floatX),
                                      borrow=True)

            self._layer_wise_autoencoders.append(
                {'object': [input_layer, dropout_layer, encoder_layer, decoder_layer],
                 'encoder_layer': encoder_layer,
                 'decoder_layer': decoder_layer})

        # Print layer wise topology
        print '> Layerwise topology'
        print '----------------------------------'
        for i in range(0, len(self._layer_wise_autoencoders)):
            print lasagne.layers.get_output_shape(self._layer_wise_autoencoders[i]['object'])
        print '----------------------------------'

    def init_deep(self, dataset, deep_epochs=100000,  filename='predec_model.zip', delete_layerwise=False, labels=None):
        # Create deep network
        input_var = T.matrix('input_var')
        index = T.lscalar()
        network = []
        input_layer = lasagne.layers.InputLayer(
            shape=(None, self._dims[0][0]), input_var=input_var)
        self.INPUT_LAYER = input_layer
        network.append(input_layer)
        for i in range(0, len(self._layer_wise_autoencoders)):
            network.append(lasagne.layers.DenseLayer(incoming=input_layer if i == 0 else network[-1],
                                                     num_units=self._dims[
                                                         i][1],
                                                     W=self._layer_wise_autoencoders[i]['encoder_layer'].W,
                                                     nonlinearity=self.enc_act[i]))
        for i in reversed(range(0, len(self._layer_wise_autoencoders))):
            network.append(lasagne.layers.DenseLayer(incoming=network[-1],
                                                     num_units=self._dims[i][
                                                         2] if i != 0 else self._feature_shape,
                                                     W=self._layer_wise_autoencoders[i]['decoder_layer'].W,
                                                     nonlinearity=self.dec_act[i]))

        self._deep_ae = {'object': network,
                         'encoder_layer': network[len(self._dims)],
                         'decoder_layer': network[-1]}

        print '> Deep neural net Topology'
        print '----------------------------------'
        print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(self._deep_ae['object']))
        print '----------------------------------'

        # Create train function
        base_lr = self.learning_rate
        X = th.shared(name='X', value=np.asarray(dataset,
                                                 dtype=th.config.floatX),
                           borrow=True)
        learning_rate = T.scalar(name='learning_rate')
        index = T.lscalar()
        prediction = lasagne.layers.get_output(self._deep_ae['decoder_layer'])
        cost = lasagne.objectives.squared_error(
            prediction, input_layer.input_var).mean()
        params = lasagne.layers.get_all_params(
            self._deep_ae['decoder_layer'], trainable=True)
        updates = lasagne.updates.momentum(
            cost, params, learning_rate=learning_rate)
        train = th.function(
            inputs=[index, learning_rate], outputs=cost, updates=updates, givens={input_layer.input_var: X[index:index + self.mini_batch_size, :]})

        # train
        print '> Deep neural net trainining'
        for epoch in xrange(deep_epochs):
            dataset = dataset.astype(np.float32)
            for row in xrange(0, dataset.shape[0], self.mini_batch_size):
                loss = train(row, base_lr)
            if epoch % 10 == 0:
                log(str(epoch) + ' ' + str(loss), label='DNT')
                # print '>'+str(datetime.datetime.utcnow())+ ' Epoch ' +
                # str(epoch) + ' Loss function value.... : ' + str(loss)
            if epoch % 100 == 0:
                if not(labels is None):
                    kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=-1)
                    self.INPUT_LAYER.input_var = X
                    hidden = lasagne.layers.get_output(
                        self._deep_ae['encoder_layer']).eval()
                    cluster_prediction = kmeans.fit_predict(hidden)
                    acc = cluster_acc(labels,cluster_prediction)
                    log(str(epoch) + ' ' + str(acc), label='DNT_ACC')
            if epoch % self.lr_epoch_decay == 0:
                base_lr = base_lr / 10
            if epoch % 1000 == 0:
                self.save(filename)
        self.save(filename)

    def target_dist(self, q):
        q = (q.T / q.sum(axis=1)).T
        p = (q**2)
        p = (p.T / p.sum(axis=1)).T
        return p

    def init_dec(self, dataset, n_clusters=10, n_init=20):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=-1)
        self.INPUT_LAYER.input_var = th.shared(name='X', value=np.asarray(dataset,
                                                                          dtype=th.config.floatX),
                                               borrow=True)
        hidden = lasagne.layers.get_output(
            self._deep_ae['encoder_layer']).eval()
        self.cluster_prediction = kmeans.fit_predict(hidden)
        self._centroids = kmeans.cluster_centers_
        c_layer = ClusteringLayer(incoming=self._deep_ae[
                                  'encoder_layer'], num_units=n_clusters,
                                  W=self._centroids,
                                  shape=(n_clusters, hidden.shape[1]))
        self._dec = self._deep_ae['object'][
            0:len(self._dims) + 1]  # TODO write better
        self._dec.append(c_layer)
        print '> DEC Topology'
        print '----------------------------------'
        print lasagne.layers.get_output_shape(self._dec)
        print '----------------------------------'


    def train_dec(self, dataset, tol=0.01,  filename='dec_model.zip', dec_epochs=100000):
        X = th.shared(name='X', value=np.asarray(dataset,
                                                 dtype=th.config.floatX),
                      borrow=True)
        self.INPUT_LAYER.input_var = X
        base_lr = 0.2
        x = T.matrix('x')
        prediction = lasagne.layers.get_output(self._dec[-1])
        x = self.target_dist(prediction)
        cost = lasagne.objectives.categorical_crossentropy(
            prediction, x).mean()
        params = lasagne.layers.get_all_params(
            self._dec, trainable=True)
        updates = lasagne.updates.momentum(
            cost, params, learning_rate=base_lr)
        train = th.function(
            inputs=[], outputs=[cost, prediction.argmax(1), prediction],
            updates=updates, givens={self.INPUT_LAYER.input_var: X})
        self._DEC = {'object': self._dec, 'train': train}
        print '> Cluster finetuning'
        train = True
        epochs = 0
        while train:
            loss, new_pred, prediction = self._DEC[
                'train']()
            delta = ((new_pred != self.cluster_prediction).sum().astype(
                np.float32) / new_pred.shape[0])
            print new_pred
            if (epochs > dec_epochs) or (delta < tol):
                train = False
                self.save(filename)
            else:
                self.cluster_prediction = new_pred
            if epochs % 10 == 0:
                # print '> Epoch ' + str(epochs) + ' Change in label
                # assignment.... : ' + str(delta)
                log(str(epochs) + ' ' + str(loss) + ' ' + str(delta),
                label='DEC')
            if epochs % 1000 == 0:
                self.save(filename)
            epochs += 1


    def get_hidden(self, dataset):
        self.INPUT_LAYER.input_var = dataset
        return lasagne.layers.get_output(self._deep_ae['encoder_layer']).eval()

    def save(self, filename='dec_model.zip'):
        utils.save(filename, self)

    def load(self, filename='dec_model.zip'):
        self = utils.load_single(filename)

    def save_cpu(self):
        self.INPUT_LAYER.input_var = None
        lw = []
        if hasattr(self, '_layer_wise_autoencoders'):
            for i in range(0, len(self._layer_wise_autoencoders)):
                lw.append(lasagne.layers.get_all_param_values(
                    self._layer_wise_autoencoders[i]['object']))
            utils.save('layer_wise_cpu.zip', lw)
        if hasattr(self, '_deep_ae'):
            deep = lasagne.layers.get_all_param_values(self._deep_ae['object'])
            utils.save('deep_ae_cpu.zip', deep)
        if hasattr(self, '_DEC'):
            dec = lasagne.layers.get_all_param_values(self._DEC['object'])
            utils.save('dec_ae_cpu.zip', dec)
