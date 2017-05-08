import numpy as np
import lasagne
import theano
import dataset_utils as utils

class Model(object):

  def __init__(self, input_layer, encoder_layer, decoder_layer, network):
      self._input_layer = input_layer
      self._encoder_layer = encoder_layer
      self._decoder_layer = decoder_layer
      self._network = network


  def get_hidden(self, dataset):
      self._input_layer.input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                       dtype=theano.config.floatX),
                                    borrow=True)
      hidden = lasagne.layers.get_output(self._encoder_layer).eval()
      return hidden


  def get_output(self, dataset):
      self._input_layer.input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                       dtype=theano.config.floatX),
                                    borrow=True)
      output = lasagne.layers.get_output(self._network).eval()
      return output

  def save(self,filename='model_template.zip'):
      utils.save(filename,self)

  def load(self,filename='model_template.zip'):
      self = utils.load_single(filename)
