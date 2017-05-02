#!/usr/bin/env python

import sys
import os
from datetime import datetime
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES,exception_verbosity=high'
# os.environ['THEANO_FLAGS'] = 'device=cpu'
from netcdf_subset import netCDF_subset
from Dataset_transformations import Dataset_transformations
import numpy as np
from sda import sda
import dataset_utils as utils
import lasagne
import theano as th
import time
import pickle
from disputil import displayz

def log(s, label='INFO'):
    sys.stderr.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')

log('Loading MNIST...')
[X,labels] = utils.load_mnist(path='/mnt/disk1/thanasis/autoencoder')
log('Done')


samples_slice = slice(0, 1000)
X = X[samples_slice,:]
labels = labels[samples_slice, :]
print X.shape
print labels.shape
# Sda = utils.load_single('predec_model.zip')

#

Sda = sda(feature_shape=784,
          dims=[(784,500,784),(500,500,500),(500,2000,500)], #,500,2000,10],
          learning_rate=0.1,
          lr_epoch_decay=2000,
          mini_batch_size=100,
          corruption_factor=0.2)

# Sda2 = sda(feature_shape=500,
#           dims=[(500,500,500)], #,500,2000,10],
#           learning_rate=0.1,
#           lr_epoch_decay=2000,
#           mini_batch_size=1000,
#           corruption_factor=0.2)

Sda.init_lw(X,lw_epochs=5000,filename='layerwise_models_784.zip')
# Sda._layer_wise_autoencoders[0]['object'][0].input_var = th.shared(name='X', value=np.asarray(X,
#                                               dtype=th.config.floatX),
#                   borrow=True)
# hidden = lasagne.layers.get_output(Sda._layer_wise_autoencoders[0]['encoder_layer']).eval()
# Sda2.init_lw(hidden,lw_epochs=100)


#
# Sda.init_deep(X,deep_epochs=10,filename='predec_model_100.zip',labels=labels)
#
#
# Sda.init_dec(X, n_clusters=10)
# Sda.train_dec(X,filename='dec_model_100.zip')
