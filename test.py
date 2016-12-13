import datetime
import numpy as np
from scipy.cluster.hierarchy import fcluster, dendrogram
from netcdf_subset import netCDF_subset
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans
import cPickle
from matplotlib.mlab import PCA as mlabPCA
from Kclustering import Kclustering
from Hclustering import Hclustering

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    dsin = Dataset(inp, "r")
    data_dict = {'dataset': dsin, 'levels': [500],
                 'sub_vars': ['GHT'], 'lvlname': 'num_metgrid_levels',
                 'timename': 'Times', 'time_unit': 'hours since 1900-01-01 00:00:0.0',
                 'time_cal': 'gregorian', 'ncar_lvls': None}
    c_dict = {'n_clusters':6,'normalize':True,'affinity':'euclidean','method':'ward','season':None,'therm_season':None,'multilevel':False,'size_desc':24,'size_div':3}
    kc = Hclustering(c_dict,data_dict)
    kc.link_var()
