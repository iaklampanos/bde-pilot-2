import numpy as np
import theano as th
import theano.tensor as T
from lasagne.regularization import regularize_layer_params_weighted, l2
from lasagne.regularization import regularize_layer_params
import lasagne
from ClusteringLayer import ClusteringLayer
from sklearn.cluster import KMeans
import gzip
import cPickle
import dataset_utils as utils


class sda(object):

    def __init__(self, feature_shape, encoder_dims, learning_rate,  mini_batch_size=1000, corruption_factor=0.3):
        self._feature_shape = feature_shape
        self._encoder_dims = encoder_dims
        self.learning_rate = learning_rate
        self._corruption_factor = corruption_factor
        self.INPUT_LAYER = lasagne.layers.InputLayer(
            shape=(None, feature_shape))
        self._encoders = []
        self._decoders = []
        # Init activations
        self.enc_act = []
        for i in range(0, len(self._encoder_dims)):
            if i == len(self._encoder_dims) - 1:
                # self.enc_act.append(None)
                self.enc_act.append(lasagne.nonlinearities.rectify)
            else:
                # self.enc_act.append(lasagne.nonlinearities.rectify)
                self.enc_act.append(lasagne.nonlinearities.rectify)
        self.dec_act = []
        for i in range(0, len(self._encoder_dims)):
            if i == 0:
                # self.dec_act.append(lasagne.nonlinearities.rectify)
                self.dec_act.append(lasagne.nonlinearities.sigmoid)
            else:
                # self.dec_act.append(None)
                self.dec_act.append(lasagne.nonlinearities.sigmoid)
        self._layer_wise_autoencoders = []
        self.mini_batch_size = mini_batch_size

    def init_lw(self):
        # Init layer wise
        for i in range(0, len(self._encoder_dims)):
            # Init first autoencoder
            if i == 0:
                dropout_layer = lasagne.layers.DropoutLayer(incoming=self.INPUT_LAYER,
                                                            p=self._corruption_factor)
                encoder_layer = lasagne.layers.DenseLayer(incoming=dropout_layer,
                                                          num_units=self._encoder_dims[
                                                              i],
                                                          W=lasagne.init.Normal(),
                                                          nonlinearity=self.enc_act[i])
                decoder_layer = lasagne.layers.DenseLayer(incoming=encoder_layer,
                                                          num_units=self._feature_shape,
                                                          W=lasagne.init.Normal(),
                                                          nonlinearity=self.dec_act[i])
                self._layer_wise_autoencoders.append(
                    {'object': [dropout_layer, encoder_layer, decoder_layer],
                     'dropout': dropout_layer, 'encoder_layer': encoder_layer,
                     'decoder_layer': decoder_layer})
            else:
                # Init the rest of autoencoders
                encoder_layer = lasagne.layers.DenseLayer(incoming=self._layer_wise_autoencoders[i - 1]['encoder_layer'],
                                                          num_units=self._encoder_dims[
                                                              i],
                                                          W=lasagne.init.Normal(),
                                                          nonlinearity=self.enc_act[i])
                decoder_layer = lasagne.layers.DenseLayer(incoming=encoder_layer,
                                                          num_units=self._encoder_dims[
                                                              i - 1],
                                                          W=lasagne.init.Normal(),
                                                          nonlinearity=self.dec_act[i])
                self._layer_wise_autoencoders.append(
                    {'object': [self._layer_wise_autoencoders[i - 1]['encoder_layer'], encoder_layer, decoder_layer],
                     'dropout': dropout_layer, 'encoder_layer': encoder_layer,
                     'decoder_layer': decoder_layer})

        # Print layer wise topology
        print '> Layerwise topology'
        print '----------------------------------'
        for i in range(0, len(self._layer_wise_autoencoders)):
            print lasagne.layers.get_output_shape(self._layer_wise_autoencoders[i]['object'])
        print '----------------------------------'

    def train_lw(self, dataset, filename='layerwise_models.zip', lw_epochs=50000):
        X = th.shared(name='X', value=np.asarray(dataset,
                                                 dtype=th.config.floatX),
                           borrow=True)
        # Initialize training functions
        for i in range(0, len(self._encoder_dims)):
            if i == 0:
                x = T.matrix('x')
                index = T.lscalar()
                learning_rate = T.scalar(name='learning_rate')
                prediction = lasagne.layers.get_output(
                    self._layer_wise_autoencoders[i]['decoder_layer'])
                cost = lasagne.objectives.squared_error(prediction, x).mean()
                l2_penalty = regularize_layer_params(self._layer_wise_autoencoders[
                                                     i]['object'], l2) * (0.001) / 2
                cost = cost + l2_penalty
                params = lasagne.layers.get_all_params(self._layer_wise_autoencoders[
                                                       i]['object'], trainable=True)
                updates = lasagne.updates.sgd(
                    cost, params, learning_rate=learning_rate)
                train = th.function(
                    inputs=[index, learning_rate], outputs=cost, updates=updates, givens={x: X[index:index + self.mini_batch_size, :], self.INPUT_LAYER.input_var: X[index:index + self.mini_batch_size]})
                self._layer_wise_autoencoders[i]['train'] = train
            else:
                index = T.lscalar()
                # x = T.matrix('x')
                learning_rate = T.scalar(name='learning_rate')
                x = lasagne.layers.get_output(self._layer_wise_autoencoders[
                                              i - 1]['encoder_layer'])
                prediction = lasagne.layers.get_output(
                    self._layer_wise_autoencoders[i]['decoder_layer'])
                cost = lasagne.objectives.squared_error(prediction, x).mean()
                l2_penalty = regularize_layer_params(self._layer_wise_autoencoders[
                                                     i]['object'], l2) * (0.001) / 2
                cost = cost + l2_penalty
                params = lasagne.layers.get_all_params(self._layer_wise_autoencoders[
                                                       i]['object'], trainable=True)
                updates = lasagne.updates.sgd(
                    cost, params, learning_rate=learning_rate)
                train = th.function(
                    inputs=[index, learning_rate], outputs=cost, updates=updates, givens={self.INPUT_LAYER.input_var: X[index:index + self.mini_batch_size]})
                self._layer_wise_autoencoders[i]['train'] = train

        # Start training
        print '> Layer wise train'
        for i in range(0, len(self._layer_wise_autoencoders)):
            base_lr = self.learning_rate
            print '> Autoenconder number : ' + str(i + 1)
            if i == 0:
                for epoch in xrange(lw_epochs):
                    dataset = dataset.astype(np.float32)
                    for row in xrange(0, dataset.shape[0], self.mini_batch_size):
                        loss = self._layer_wise_autoencoders[i][
                            'train'](row, base_lr)
                    if epoch % 100 == 0:
                        print '> Epoch ' + str(epoch) + ' Loss function value.... : ' + str(loss)
                    elif epoch % 20000 == 0:
                        base_lr = base_lr / 10
                    elif epoch % 1000 == 0:
                        self.save(filename)
            else:
                for epoch in xrange(lw_epochs):
                    for row in xrange(0, dataset.shape[0], self.mini_batch_size):
                        loss = self._layer_wise_autoencoders[i][
                            'train'](row, base_lr)
                    if epoch % 100 == 0:
                        print '> Epoch ' + str(epoch) + ' Loss function value.... : ' + str(loss)
                    elif epoch % 20000 == 0:
                        base_lr = base_lr / 10
                    elif epoch % 1000 == 0:
                        self.save(filename)

    def init_deep(self):
        nnet_object = []
        for i in range(0, len(self._layer_wise_autoencoders)):
            if i == 0:
                nnet_object.append(self.INPUT_LAYER)
                nnet_object.append(self._layer_wise_autoencoders[
                                   i]['encoder_layer'])
            else:
                nnet_object.append(
                    self._layer_wise_autoencoders[i]['encoder_layer'])
        for i in reversed(range(0, len(self._layer_wise_autoencoders))):
            nnet_object.append(self._layer_wise_autoencoders[
                               i]['decoder_layer'])
        self._deep_ae = {'object': nnet_object,
                         'encoder_layer': nnet_object[len(self._encoder_dims)], 'decoder_layer': nnet_object[len(nnet_object) - 1]}
        print '> Deep neural net Topology'
        print '----------------------------------'
        print lasagne.layers.get_output_shape(self._deep_ae['object'])
        print '----------------------------------'

    def train_deep(self, dataset, deep_epochs=100000,  filename='predec_model.zip', delete_layerwise=False):
        base_lr = self.learning_rate
        X = th.shared(name='X', value=np.asarray(dataset,
                                                 dtype=th.config.floatX),
                           borrow=True)
        learning_rate = T.scalar(name='learning_rate')
        index = T.lscalar()
        x = T.matrix('x')
        prediction = lasagne.layers.get_output(self._deep_ae['decoder_layer'])
        cost = lasagne.objectives.squared_error(prediction, x).mean()
        params = lasagne.layers.get_all_params(
            self._deep_ae['decoder_layer'], trainable=True)
        updates = lasagne.updates.sgd(
            cost, params, learning_rate=learning_rate)
        train = th.function(
            inputs=[index, learning_rate], outputs=cost, updates=updates, givens={x: X[index:index + self.mini_batch_size, :], self.INPUT_LAYER.input_var: X[index:index + self.mini_batch_size]})
        self._deep_ae['train'] = train
        # Start training
        print '> Deep neural net trainining'
        for epoch in xrange(deep_epochs):
            dataset = dataset.astype(np.float32)
            for row in xrange(0, dataset.shape[0], self.mini_batch_size):
                loss = self._deep_ae[
                    'train'](row, base_lr)
            if epoch % 100 == 0:
                print '> Epoch ' + str(epoch) + ' Loss function value.... : ' + str(loss)
            elif epoch % 20000 == 0:
                base_lr = base_lr / 10
            elif epoch % 1000 == 0:
                self.save(filename)
        self.save(filename)
        if delete_layerwise:
            del self._layer_wise_autoencoders

    def target_dist(self, q):
        weight = q**2 / q.sum()
        return (weight.T / weight.sum()).T

    def init_dec(self, dataset, n_clusters=10, n_init=100):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=-1)
        self.INPUT_LAYER.input_var = th.shared(name='X', value=np.asarray(dataset,
                                                                          dtype=th.config.floatX),
                                               borrow=True)
        hidden = lasagne.layers.get_output(
            self._deep_ae['encoder_layer']).eval()
        self.cluster_prediction = kmeans.fit_predict(hidden)
        self._centroids = kmeans.cluster_centers_
        c_layer = ClusteringLayer(incoming=self._deep_ae[
                                  'encoder_layer'], num_units=n_clusters, W=self._centroids, shape=(n_clusters, hidden.shape[1]))
        self._dec = [self._deep_ae['encoder_layer'], c_layer]
        print '> DEC Topology'
        print '----------------------------------'
        print lasagne.layers.get_output_shape(self._dec)
        print '----------------------------------'
        return [self._centroids, self._deep_ae['encoder_layer'].W]

    def train_dec(self, dataset, tol=0.1,  filename='dec_model.zip', dec_epochs=100000):
        self.INPUT_LAYER.input_var = th.shared(name='X', value=np.asarray(dataset,
                                                                          dtype=th.config.floatX),
                                               borrow=True)
        base_lr = self.learning_rate
        learning_rate = T.scalar(name='learning_rate')
        x = T.matrix('x')
        prediction = lasagne.layers.get_output(self._dec[len(self._dec) - 1])
        x = self.target_dist(prediction)
        cost = lasagne.objectives.categorical_crossentropy(
            prediction, x).mean()
        params = lasagne.layers.get_all_params(
            self._dec, trainable=True)
        updates = lasagne.updates.sgd(
            cost, params, learning_rate=learning_rate)
        train = th.function(
            inputs=[learning_rate], outputs=[cost, prediction.argmax(1)], updates=updates)
        self._dec = {'object': self._dec, 'train': train}
        print '> Cluster finetuning'
        train = True
        epochs = 0
        while train:
            dataset = dataset.astype(np.float32)
            loss, new_pred = self._dec[
                'train'](base_lr)
            delta = ((new_pred == self.cluster_prediction).sum().astype(
                np.float32) / new_pred.shape[0])
            if (epochs > dec_epochs) or (delta < tol):
                train = False
                self.save(filename)
            if epochs % 100 == 0:
                print '> Epoch ' + str(epochs) + ' Change in label assignment.... : ' + str(delta)
            elif epochs % 20000 == 0:
                base_lr = base_lr / 10
            elif epochs % 1000 == 0:
                self.save(filename)
            epochs += 1
        return [lasagne.layers.get_all_param_values(self._dec['object'][1])[0], self._dec['object'][len(self._dec) - 1].W]

    def get_hidden(self, dataset):
        self.INPUT_LAYER.input_var = dataset
        return lasagne.layers.get_output(self._dec['obect'][0])

    def save(self, filename='dec_model.zip'):
        self.INPUT_LAYER.input_var = None
        utils.save(filename, self)

    def load(self, filename='dec_model.zip'):
        self = utils.load_single(filename)


    # def save_model_cpu(self,filename):
