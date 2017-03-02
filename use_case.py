from netcdf_subset import netCDF_subset
from operator import attrgetter
from argparse import ArgumentParser
from Dataset_transformations import Dataset_transformations
from Dataset import Dataset
from Clustering import Clustering
import numpy as np
from sklearn.cluster import KMeans
import dataset_utils as utils
from ClusteringExperiment import ClusteringExperiment
from Autoencoder import AutoEncoder
from theano import tensor as T

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    data_dict = netCDF_subset(inp, [500], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
    #data_dict2 = netCDF_subset(inp, [1000], ['TT'], lvlname='num_metgrid_levels', timename='Times')
    export_template = netCDF_subset('/mnt/disk1/thanasis/data/train.nc', [500], ['UU'], lvlname='num_metgrid_levels', timename='Times')
    items = [data_dict.extract_data()]
    items = np.array(items)
    # #items = items[:,:,:,:,range(0,32),:]
    #items = items[:,:,:,:,:,range(0,32)]
    print items.shape
    ds = Dataset_transformations(items,1000,items.shape)
    ds.twod_transformation()
    #ds.normalize()
    #ds.shift()
    print np.min(ds._items),np.max(ds._items)
    #data = ds.get_items()
    #print data.shape
    #data = data[,:]
    #ds._items = data
    #print ds._items.shape
    clust_obj = Clustering(ds,n_clusters=3,n_init=100,features_first=True)
    A = AutoEncoder(X=np.transpose(ds.get_items()), hidden_size=3000,
                     activation_function=T.nnet.sigmoid,
                     output_function=T.nnet.sigmoid,
                     n_epochs=400, mini_batch_size=1000,
                     sparsity_level=0.05, sparse_reg=1e-4,
                     learning_rate=0.3,
                     corruption_level=0.4,
                     corrupt=True
                     )
    exper = ClusteringExperiment(ds,A,clust_obj)
    exper.start()
    exper.plot_output_frames(64,64)
    #clust_obj.kmeans()
    #clust_obj.create_descriptors(13)
    #print np.array(clust_obj._descriptors).shape
    #utils.export_descriptor_kmeans(outp,export_template,clust_obj)
