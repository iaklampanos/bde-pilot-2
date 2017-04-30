import numpy as np
import theano as th
import theano.tensor as T

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
        q = 1.0 / (1.0 + T.sqrt((T.sqr(expand_dims(input, 1) -
                                          self.W).sum(axis=2)))**2 / self.alpha)
        q = q**((self.alpha + 1.0) / 2.0000)
        q = (q.T / q.sum(axis=1)).T
        return q

    # def get_all_param_values(self):
    #     return self.W.eval()

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)
