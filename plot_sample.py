import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
from sda import sda
import numpy as np
import lasagne
import theano as th
import dataset_utils as utils
from sklearn.cluster import KMeans
from sklearn import metrics

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in xrange(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size




[X,labels] = utils.load_mnist(dataset='testing',path='/mnt/disk1/thanasis/autoencoder/')
print X.shape
X = X[0:1000,:]
labels = labels[0:1000,:].reshape(1000)
Sda = utils.load_single('layerwise_models_784.zip')

#Sda.INPUT_LAYER.input_var =
Sda._layer_wise_autoencoders[0]['object'][0].input_var = th.shared(name='X', value=np.asarray(X,
                                              dtype=th.config.floatX),
                   borrow=True)
# a_out =  lasagne.layers.get_output(Sda._layer_wise_autoencoders[0]['decoder_layer']).eval()

# a_out =  lasagne.layers.get_output(Sda._layer_wise_autoencoders[0]['decoder_layer']).eval()

# kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=-1)
#
#
# hidden = lasagne.layers.get_output(Sda._deep_ae['encoder_layer']).eval()
# cluster_prediction = kmeans.fit_predict(hidden)
# acc = cluster_acc(labels,cluster_prediction)
#
# print 'DEEP',acc
#
kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=-1)


hidden = lasagne.layers.get_output(Sda._layer_wise_autoencoders[0]['encoder_layer']).eval()
cluster_prediction = kmeans.fit_predict(hidden)
acc = cluster_acc(cluster_prediction,labels)

print 'LW',acc

for i in range(0,100):
    utils.plot_pixel_image(X[i,:],a_out[i,:],28,28)
