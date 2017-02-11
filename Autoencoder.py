import numpy as np
from theano import tensor as T
import theano as th
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from Clustering import Clustering
from numpy import random as rng

class AutoEncoder(object):

    def __init__(self, X, hidden_size, activation_function,
                 output_function, corrupt=False, n_epochs=100, mini_batch_size=1,
                 learning_rate=0.1, sparsity_level=0.01, sparse_reg=1,
                 corruption_level=0.):
        assert type(X) is np.ndarray
        assert len(X.shape) == 2
        self.input = X
        self.X = X
        self.X = th.shared(name='X', value=np.asarray(self.X,
                                                      dtype=th.config.floatX),
                           borrow=True)
        self.n = X.shape[1]
        self.m = X.shape[0]
        assert type(hidden_size) is int
        assert hidden_size > 0
        self.hidden_size = hidden_size
        initial_W = np.asarray(rng.uniform(
            low=-4 * np.sqrt(6. / (self.hidden_size + self.n)),
            high=4 * np.sqrt(6. / (self.hidden_size + self.n)),
            size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                                                      dtype=th.config.floatX),
                            borrow=True)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,),
                                                      dtype=th.config.floatX),
                            borrow=True)
        self.activation_function = activation_function
        self.output_function = output_function
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.sparsity_level = sparsity_level
        self.sparse_reg = sparse_reg
        self.corruption_level = corruption_level
        self.corrupt = corrupt

    def kl_divergence(self, p, p_hat):
        eps = 0
        term1 = p + eps * T.log(p + eps)
        term2 = p + eps * T.log(p_hat + eps)
        term3 = (1 - p + eps) * T.log(1 - p + eps)
        term4 = (1 - p + eps) * T.log(1 - p_hat + eps)
        return term1 - term2 + term3 - term4

    def sparsity_penalty(self, h, sparsity_level=0.01, sparse_reg=1):
        sparsity_penalty = 0
        avg_act = h.mean(axis=0)
        kl_div = self.kl_divergence(sparsity_level, avg_act)
        sparsity_penalty = sparse_reg * kl_div.sum()
        return sparsity_penalty

    def get_corrupted_input(self, input, corruption_level):
        return RandomStreams(np.random.RandomState().randint(2 ** 30)).binomial(size=input.shape, n=1,
                                                                                p=1 - corruption_level,
                                                                                dtype=th.config.floatX) * input

    def train(self):
        index = T.lscalar()
        x = T.matrix('x')
        if self.corrupt:
            x = self.get_corrupted_input(x, self.corruption_level)
        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(T.dot(x, self.W) + self.b1)
        output = T.dot(hidden, T.transpose(self.W)) + self.b2
        output = self.output_function(output)
        L = T.mean(((output - x)**2).sum(axis=1))
        L1 = (self.W ** 2).sum()
        Spars = self.sparsity_penalty(
            hidden, self.sparsity_level, self.sparse_reg)
        cost = L + (0.001) / 2 * L1 + Spars
        updates = []
        gparams = T.grad(cost, params)
        for param, gparam in zip(params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x: self.X[index:index + self.mini_batch_size, :]})

        print gparam
        start_time = time.clock()
        for epoch in xrange(self.n_epochs):
            print "Epoch:", epoch
            ccost = []
            for row in xrange(0, self.m, self.mini_batch_size):
                c = train(row)
                ccost.append(c[0])
            print np.mean(ccost)
        end_time = time.clock()
        print "Average time per epoch=", (end_time - start_time) / self.n_epochs
        self.hidden = self.get_hidden(self.input)
        self.decode = self.get_output(self.input)

    def get_hidden(self, data):
        x = T.dmatrix('x')
        hidden = self.activation_function(T.dot(x, self.W) + self.b1)
        transformed_data = th.function(inputs=[x], outputs=[hidden])
        return np.array(transformed_data(data)).reshape(self.m, self.hidden_size)

    def get_output(self, data):
        x = T.dmatrix('x')
        output = self.output_function(T.dot(self.activation_function(
            T.dot(x, self.W) + self.b1), T.transpose(self.W)) + self.b2)
        transformed_data = th.function(inputs=[x], outputs=[output])
        return np.array(transformed_data(data)).reshape(self.m, self.n)

    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]


def load_weather_data(ncobj,time_idx=None,normalize=True):
    if time_idx is None:
        var_list = ncobj.extract_data(ncobj.lvl_pos())
    else:
        var_list = ncobj.extract_timeslotdata(time_idx, ncobj.lvl_pos())
    if not multivar:
        for v in var_list:
            # create place holder variable where the grid is flattened
            var_data = np.ndarray(
                shape=(v.shape[0], v[0][:].flatten().shape[0]))
            for i in range(0, v.shape[0]):
                var_data[i] = v[i][:].flatten()
            print var_data.shape
        if normalize:
            for j in range(0, uv.shape[0]):
                mean = uv[j, :].mean()
                uv[j, :] = np.subtract(uv[j, :], mean)
                uv[j, :] = np.divide(uv[j, :], np.std(uv[j, :]))
    else:
        uv = Clustering.preprocess_multivar(var_list)
        # for normalization purposes we get the column mean and subtract it
        if normalize:
            for j in range(0, uv.shape[0]):
                mean = uv[j, :].mean()
                uv[j, :] = np.subtract(uv[j, :], mean)
                uv[j, :] = np.divide(uv[j, :], np.std(uv[j, :]))
    return uv


def setup_autoencoder(dataset, hidden_size=100, activation_function=T.nnet.sigmoid,
                      output_function=T.nnet.sigmoid, n_epochs=100,
                      mini_batch_size=1, corrupt=True,train=False):
    A = AutoEncoder(X=dataset, hidden_size=hidden_size,
                    activation_function=activation_function,
                    output_function=output_function,
                    n_epochs=n_epochs, mini_batch_size=mini_batch_size,
                    corrupt=corrupt
                    )
    if train:
        A.train()
    return A
