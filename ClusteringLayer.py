import numpy as np
import theano as th
import theano.tensor as T
from scipy.spatial import distance
import lasagne

def expand_dims(x, axis=-1):
    # Keras back end theano
    pattern = [i for i in range(x.type.ndim)]
    if axis < 0:
        if x.type.ndim == 0:
            axis = 0
        else:
            axis = axis % x.type.ndim + 1
    pattern.insert(axis, 'x')
    y = x.dimshuffle(pattern)
    if hasattr(x, '_keras_shape'):
        shape = list(x._keras_shape)
        shape.insert(axis, 1)
        y._keras_shape = tuple(shape)
    return y

class ClusteringLayer(lasagne.layers.Layer):

    def __init__(self,incoming,num_units,W,shape,alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(incoming=incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.alpha = alpha
        self.W = self.add_param(W,shape,name='W',trainable=True)

    def get_output_for(self,input, **kwargs):
        # p = 1.0
        # squared_euclidean_distances = (input ** 2).sum(1).reshape((input.shape[0], 1)) + (self.W ** 2).sum(1).reshape((1, self.W.shape[0])) - 2 * input.dot(self.W.T)
        # dist = T.sqrt(squared_euclidean_distances)
        # q = 1.0/(1.0 + dist**2 / self.alpha)**((self.alpha + p)/2.0)
        # q = (q.T/q.sum(axis=1)).T
        # return q
        q = 1.0/(1.0 + T.sqrt(T.sum(T.square(expand_dims(input, 1) - self.W), axis=2))**2 /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = T.transpose(T.transpose(q)/T.sum(q, axis=1))
        return q

    # def get_all_param_values(self):
    #     return self.W.eval()

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)
