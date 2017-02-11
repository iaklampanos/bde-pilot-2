import datetime
import numpy as np
from scipy.cluster.hierarchy import fcluster, dendrogram
from netcdf_subset import netCDF_subset, ncar_to_ecmwf_type
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans,MiniBatchKMeans
import pickle
from matplotlib.mlab import PCA as mlabPCA
from Kclustering import Kclustering
from Hclustering import Hclustering
from Meta_clustering import Meta_clustering
from sklearn import metrics
from Autoencoder import AutoEncoder,setup_autoencoder


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    #dsin = Dataset(inp, "r")
    #data_dict = {'dataset': dsin, 'levels': [500, 700, 900],
    #             'sub_vars': ['UU', 'VV'], 'lvlname': 'num_metgrid_levels',
    #             'timename': 'Times', 'time_unit': 'hours since 1900-01-01 00:00:0.0',
    #             'time_cal': 'gregorian', 'ncar_lvls': None}
    UVT = np.load('q.npy')
    print UVT.shape
    clust = []
    CH = []
    print UVT.shape
    A = setup_autoencoder(dataset=UVT,hidden_size=100,mini_batch_size=100,n_epochs=100,train=True,corrupt=True)
    exit(-1)
    #with open('weather_eval/autoencoders_4000.pkl', 'wb') as output:
    #    pickle.dump(A, output, pickle.HIGHEST_PROTOCOL)
    UVT = A.encode(UVT)
    V = MiniBatchKMeans(n_clusters=14,n_init=20,
               n_jobs=-1).fit(UVT).labels_
    clust.append(V)
    CH.append(metrics.calinski_harabaz_score(UVT, V))
    clust = np.array(clust)
    CH = np.array(CH)
    np.save('weather_eval/clusters.npy', clust)
    np.save('weather_eval/CH.npy', CH)
    '''
    max_ret_list = kc._netcdf_subset.find_continuous_timeslots(clut_list)
    data_dict = {'dataset': Dataset('../../data/1986_1990.nc', 'r'), 'levels': [500,700,900],
                 'sub_vars': ['UU', 'VV'], 'lvlname': 'num_metgrid_levels',
                 'timename': 'Times', 'time_unit': 'hours since 1900-01-01 00:00:0.0',
                 'time_cal': 'gregorian', 'ncar_lvls': None}
    kc2 = Kclustering(c_dict, data_dict)
    print max_ret_list
    kc2.cluster_descriptor_middle(outp, max_ret_list)
    '''
