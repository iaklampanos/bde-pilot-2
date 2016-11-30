import numpy as np
from scipy.cluster.hierarchy import fcluster,dendrogram
from netcdf_subset import netCDF_subset,calculate_clut_metrics,calculate_prf,ncar_to_ecmwf_type
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
from sklearn.decomposition import PCA as PCA
from sklearn.cluster import KMeans
import cPickle
from matplotlib.mlab import PCA as mlabPCA

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input',required=True,type=str,
                        help='input file')
    parser.add_argument('-o', '--output',type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input','output')
    inp,outp = getter(opts)
    dsin = Dataset(inp,"r")
    dorg = Dataset('/mnt/disk1/thanasis/data/1986_1987_original.nc','r')
    level2 = [500]
    vs2 = ['GHT']
    n_sub2 = netCDF_subset(dsin,level2,vs2,'num_metgrid_levels','Times')
    clut_list,Z,c_dist = n_sub2.link_multivar(12,'kmeans')
    max_ret_list = n_sub2.find_continuous_timeslots(clut_list=clut_list)
    n_sub = netCDF_subset(dorg,level2,vs2,'num_metgrid_levels','Times')
    n_sub.cluster_descriptor_max(outp,max_ret_list)
