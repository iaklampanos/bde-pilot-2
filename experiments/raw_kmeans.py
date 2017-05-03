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
    items = np.load(inp)
    items = np.transpose(items)
    print items.shape
    ds = Dataset_transformations(items, 1000, items.shape)
    ds.normalize()
    clust_obj = Clustering(ds,n_clusters=14,n_init=100,features_first=True)
    export_template = netCDF_subset('/mnt/disk1/thanasis/data/11_train.nc', [700], [
                                    'GHT'], lvlname='num_metgrid_levels', timename='Times')
    clust_obj.desc_date(export_template)
    clust_obj.create_descriptors(14)
    utils.export_descriptor_kmeans(outp,data_dict,clust_obj)
    clust_obj.save('GHT_700_raw_kmeans.zip')
