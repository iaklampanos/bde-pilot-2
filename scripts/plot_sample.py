import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
from sda import sda
import numpy as np
import lasagne
import theano as th
import dataset_utils as utils
from sklearn.cluster import KMeans
from sklearn import metrics

from sklearn.utils.linear_assignment_ import linear_assignment
def cluster_acc(y_true, y_pred):
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max())+1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind])*1.0/y_pred.size


#[X1,labels1] = utils.load_mnist(dataset='training',path='/home/ubuntu/data/mnist/')
#[X2,labels2] = utils.load_mnist(dataset='testing',path='/home/ubuntu/data/mnist/')
#X = np.concatenate((X1,X2),axis=0)
#labels = np.concatenate((labels1,labels2),axis=0)
#labels = labels.reshape(70000)
#p = np.load('random_perm.npy')
#X = X[p]
#labels = labels[p]
#[X,labels] = utils.load_mnist(dataset='testing',path='/home/ubuntu/data/mnist/')
# print X.shape
# X = X[0:10000,:]
# labels = labels[0:10000,:].reshape(10000)
#labels = labels.reshape(10000)
#Sda = utils.load_single('predec_model.zip')

#Sda.INPUT_LAYER.input_var = Sda._layer_wise_autoencoders[0]['object'][0].input_var = th.shared(name='X', value=np.asarray(X,
#                                              dtype=th.config.floatX),
#                   borrow=True)
# a_out =  lasagne.layers.get_output(Sda._layer_wise_autoencoders[0]['decoder_layer']).eval()

#a_out =  lasagne.layers.get_output(Sda._deep_ae['decoder_layer']).eval()

#kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=-1)


#hidden = np.load('MNIST_sda_hidden.npy')
#cluster_prediction = kmeans.fit_predict(hidden)
#acc = cluster_acc(labels,cluster_prediction)
#nmi = metrics.normalized_mutual_info_score(labels,cluster_prediction)
#print 'DEEP',acc #,nmi
#
#kmeans = KMeans(n_clusters=9, n_init=20, n_jobs=-1)


#hidden = lasagne.layers.get_output(Sda._layer_wise_autoencoders[0]['encoder_layer']).eval()
#cluster_prediction = kmeans.fit_predict(hidden)
#acc = cluster_acc(cluster_prediction,labels)
#nmi = metrics.normalized_mutual_info_score(labels,cluster_prediction)
#print 'LW',acc,nmi

#kmeans = KMeans(n_clusters=9, n_init=20, n_jobs=-1)
#cluster_prediction = kmeans.fit_predict(X)
#acc = cluster_acc(cluster_prediction,labels)
#nmi = metrics.normalized_mutual_info_score(labels,cluster_prediction)
#print 'K',acc,nmi
#X = np.load('/home/ubuntu/data/GHT_700_vanilla.npy')
X = np.load('/home/ubuntu/data/GHT_700.npy')
p = np.random.permutation(X.shape[0])
X = X[p,:]
#from Dataset_transformations import Dataset_transformations
#ds = Dataset_transformations(X, 1000, X.shape)
#ds._items = np.transpose(ds._items)
#ds.normalize()
#ds._items = np.transpose(ds._items)
#np.save('/home/ubuntu/data/GHT_700.npy',ds._items)
a_out = np.load('./2600/GHT_700_0_pretrained_output.npy')
a_out = a_out[p,:]
for i in range(0,100):
    utils.plot_pixel_image(X[i,:],a_out[i,:],64,64)

