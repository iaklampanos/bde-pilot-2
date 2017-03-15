import sys
sys.path.append('..')
sys.setrecursionlimit(10000)

import os
os.environ[
    'THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'

from netcdf_subset import netCDF_subset
from operator import attrgetter
from argparse import ArgumentParser
from Dataset_transformations import Dataset_transformations
from ClusteringExperiment import ClusteringExperiment
from Clustering import Clustering
from conv_autoencoder import ConvAutoencoder
from theano import tensor as T
import dataset_utils as utils
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    data_dict = netCDF_subset(
        inp, [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
    items = [data_dict.extract_piece(range(0,11688),range(0,64),range(0,64))]
    items = np.array(items)
    print items.shape
    ds = Dataset_transformations(items, 1000, items.shape)
    ds.twod_transformation()
    ds.normalize()
    ds.shift()
    X_train, X_out = ds.conv_process(items)
    ds._items = X_train
    print X_train.shape
    print ds.get_items().shape
    CA = ConvAutoencoder(X_train=X_train, conv_filters=32, deconv_filters=32,
                         filter_sizes=3, epochs=100, hidden_size=1000, channels=1,
                         stride=2, corruption_level=0.3, l2_level=(0.001) / 2,
                         samples=100, features_x=64, features_y=64)
    clust_obj = Clustering(ds, n_clusters=14, n_init=100)
    exper = ClusteringExperiment(ds, CA, clust_obj)
    exper.start()
    exper.clustering()
    exper._clustering.create_descriptors(14)
    utils.export_descriptor_kmeans(outp,data_dict,exper._clustering)
    utils.save('Clustering_object.zip', exper._clustering)
    utils.save('CE_ConvAutoencoder_700_ght.zip', exper)
