import numpy as np
from theano import tensor as T
import theano as th
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from numpy import random as rng
import dataset_utils as utils

class AutoEncoder(object):


    def __init__(self, X, hidden_size, activation_function,
                 output_function, corrupt=False, n_epochs=100, mini_batch_size=1,
                 learning_rate=0.1, sparsity_level=0.05, sparse_reg=1e-3,
                 corruption_level=0.0):
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
        eps = 1e-4
        term1 = p + eps * T.log(p + eps)
        term2 = p + eps * T.log(p_hat + eps)
        term3 = (1 - p + eps) * T.log(1 - p + eps)
        term4 = (1 - p + eps) * T.log(1 - p_hat + eps)
        return term1 - term2 + term3 - term4

    def sparsity_penalty(self, h, sparsity_level=0.05, sparse_reg=1e-4):
        #sparsity_level = th.shared(name='sparsity_level',value=sparsity_level)
        #sparsity_level = T.extra_ops.repeat(sparsity_level, self.hidden_size)
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
        mse = T.mean(((output - x)**2).sum(axis=1))
        L2 = (self.W ** 2).sum()
        # Spars = self.sparsity_penalty(
        #     hidden, self.sparsity_level, self.sparse_reg)
        cost = mse + (0.001) / 2 * L2 #+ 1*Spars
        updates = []
        gparams = T.grad(cost, params)
        for param, gparam in zip(params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        train = th.function(inputs=[index], outputs=[cost], updates=updates,
                            givens={x: self.X[index:index + self.mini_batch_size, :]})

        print gparam
        start_time = time.clock()
        loss = []
        for epoch in xrange(self.n_epochs):
            print "Epoch:", epoch
            ccost = []
            for row in xrange(0, self.m, self.mini_batch_size):
                c = train(row)
                ccost.append(c[0])
            print np.mean(ccost)
            loss.append(np.mean(ccost))
        end_time = time.clock()
        print "Average time per epoch=", (end_time - start_time) / self.n_epochs
        self.loss = loss
        self.hidden = self.get_hidden(self.input)
        self.decoded = self.get_output(self.input)

    def get_hidden(self, data=None):
        if data is None:
            return self.hidden
        else:
            x = T.dmatrix('x')
            hidden = self.activation_function(T.dot(x, self.W) + self.b1)
            transformed_data = th.function(inputs=[x], outputs=[hidden])
            return np.array(transformed_data(data)).reshape(self.m, self.hidden_size)

    def get_output(self, data=None):
        if data is None:
            return self.decoded
        else:
            x = T.dmatrix('x')
            output = self.output_function(T.dot(self.activation_function(
                T.dot(x, self.W) + self.b1), T.transpose(self.W)) + self.b2)
            transformed_data = th.function(inputs=[x], outputs=[output])
            return np.array(transformed_data(data)).reshape(self.m, self.n)

    def test(self,data):
        self.hidden = self.get_hidden(data)
        self.decoded = self.get_output(data)

    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]


    def save(self, filename='Autoencoder.zip'):
        utils.save(filename, self)

    def load(self, filename='Autoencoder.zip'):
        self = utils.load(filename)
