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

samples_slice = slice(0, 5000)
X = X[samples_slice,:]
labels = labels[samples_slice, :]
print X.shape
print labels.shape
print X.shape
#
#
Sda = sda(feature_shape=784,
          dims=[(784,500,784)], #,500,2000,10],
          learning_rate=0.1,
          lr_epoch_decay=2000,
          mini_batch_size=256,
          corruption_factor=0.2)

Sda.init_lw(X,lw_epochs=10)


Sda.init_deep(X,deep_epochs=10)


Sda.init_dec(X, n_clusters=10)
Sda.train_dec(X)
