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
from conv_autoencoder import ConvAutoencoder

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
    #data_dict2 = netCDF_subset(inp, [1000], ['TT'], lvlname='num_metgrid_levels', timename='Times')
    export_template = netCDF_subset('/mnt/disk1/thanasis/data/train.nc', [700], [
                                    'GHT'], lvlname='num_metgrid_levels', timename='Times')
    items = [data_dict.extract_timeslotdata(range(0, 100))]
    items = np.array(items)
    # #items = items[:,:,:,:,range(0,32),:]
    #items = items[:,:,:,:,:,range(0,32)]
    print items.shape
    ds = Dataset_transformations(items, 1000, items.shape)
    ds.twod_transformation()
    ds.normalize()
    ds.shift()
    print np.min(ds._items), np.max(ds._items)
    CA = ConvAutoencoder(conv_filters=32, deconv_filters=32,
                         filter_sizes=7, epochs=20, hidden_size=1000, channels=1,
                         corruption_level=0.3, l2_level=(0.001) / 2,
                         samples=100, features_x=64, features_y=64)
    X_train = np.transpose(ds.get_items())
    X_train = X_train.reshape(items.shape[2],items.shape[0],items.shape[4],items.shape[5])
    X_train = X_train.astype(np.float32)
    X_out = X_train.reshape((X_train.shape[0], -1))
    print X_train.shape
    CA.train(X_train,X_out)
    out = CA.test(X_train,X_train.shape)
    utils.plot_pixel_image(np.transpose(ds.get_items)[0],out[0],64,64)
    #data = ds.get_items()
    # print data.shape
    # data = data[,:]
    #ds._items = data
    # print ds._items.shape
    # clust_obj = Clustering(ds,n_clusters=14,n_init=100,features_first=True)
    # A = AutoEncoder(X=np.transpose(ds.get_items()), hidden_size=1000,
    #                  activation_function=T.nnet.sigmoid,
    #                  output_function=T.nnet.sigmoid,
    #                  n_epochs=100, mini_batch_size=1000,
    #                  learning_rate=0.1,
    #                  corruption_level=0.3,
    #                  corrupt=True
    #                  )
    # exper = ClusteringExperiment(ds,A,clust_obj)
    # exper.start()
    # exper.plot_output_frames(64,64)
    # clust_obj.kmeans()
    # clust_obj.create_descriptors(14)
    # print np.array(clust_obj._descriptors).shape
    # utils.export_descriptor_kmeans(outp,export_template,clust_obj)
    # clust_obj.save()
