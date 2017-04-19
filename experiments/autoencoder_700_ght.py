import sys
sys.path.append('..')

import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'

from netcdf_subset import netCDF_subset
from operator import attrgetter
from argparse import ArgumentParser
from Dataset_transformations import Dataset_transformations
from ClusteringExperiment import ClusteringExperiment
from Clustering import Clustering
from Autoencoder import AutoEncoder
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
    ds = Dataset_transformations(items, 1000, items.shape)
    ds.twod_transformation()
    ds.normalize()
    ds.shift()
    clust_obj = Clustering(ds,n_clusters=14,n_init=100,features_first=True)
    A = AutoEncoder(X=np.transpose(ds.get_items()), hidden_size=1000,
                     activation_function=T.nnet.sigmoid,
                     output_function=T.nnet.sigmoid,
                     n_epochs=100, mini_batch_size=1000,
                     learning_rate=0.1,
                     corruption_level=0.3,
                     corrupt=True
                     )
    exper = ClusteringExperiment(ds,A,clust_obj)
    exper.start()
    exper.clustering()
    exper._clustering.create_descriptors(14)
    utils.export_descriptor_kmeans(outp,data_dict,exper._clustering)
    #### add desc_date #####
    utils.save('CE_Autoencoder_700_ght.zip',exper)
