import numpy as np
from theano import tensor as T
import theano as th


class Finetune(object):

    def __init__(self, hidden, cluster_centers, alpha=1.0):
        self.X = hidden
        self.X = th.shared(name='X', value=np.asarray(self.X,
                                                      dtype=th.config.floatX),
                           borrow=True)
        self._hidden = hidden
        self.m = self._hidden.shape[0]
        self._centers = cluster_centers
        self.alpha = alpha
        self.n_epochs = 100
        print self._centers.shape
        initial_W = self._centers
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.learning_rate=0.1
        self.mini_batch_size=1000

    def stud_dist(self, index):
        q = 1.0 / (1.0 + T.sqrt((T.square(index -
                                          self.W).sum(axis=1)))**2 / self.alpha)
        q = q**((self.alpha + 1.0) / 2.0)
        q = (q.T / q.sum(axis=0)).T
        return q

    def target_dist(self, q):
        weight = q**2 / q.sum()
        return (weight.T / weight.sum()).T

    def finetune(self):
        print '------------------------------------------'
        print self._centers
        print '------------------------------------------'
        index = T.lscalar()
        x = T.matrix('x')
        q = self.stud_dist(x)
        p = self.target_dist(q)
        cost = T.sum(p * T.log(p / q + 1e-12))
        params = [self.W]
        updates = []
        gparams = T.grad(cost, params)
        for param, gparam in zip(params, gparams):
            updates.append((param, param - self.learning_rate * gparam))
        tune = th.function(inputs=[index], outputs=[cost], updates=updates,
                           givens={x: self.X[index:index + self.mini_batch_size, :]})
        for epoch in xrange(self.n_epochs):
            print "Epoch:", epoch
            for row in xrange(0, self.m, self.mini_batch_size):
                tune(row)
        print '------------------------------------------'
        print self._centers
        print ''------------------------------------------''
