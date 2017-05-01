import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
from sda import sda
import numpy as np
import lasagne
import theano as th
import dataset_utils as utils

[X,labels] = utils.load_mnist(dataset='testing',path='/mnt/disk1/thanasis/autoencoder/')
X = X[0:100,:]
Sda = utils.load_single('predec_model.zip')

# Sda.INPUT_LAYER.input_var = th.shared(name='X', value=np.asarray(X,
#                                               dtype=th.config.floatX),
#                    borrow=True)
# a_out =  lasagne.layers.get_output(Sda._layer_wise_autoencoders[0]['decoder_layer']).eval()

# a_out =  lasagne.layers.get_output(Sda._deep_ae['decoder_layer']).eval()
#
# for i in range(0,100):
#     utils.plot_pixel_image(X[i,:],a_out[i,:],28,28)
