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
import dataset_utils as utils
import json
import datetime
from netCDF4 import Dataset
import scipy
import cPickle

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    with open(inp) as data_file:
        parameters = json.load(data_file)
    test_dict = netCDF_subset(parameters['test_netcdf_path']+parameters['test_netcdf'], [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
    export_template = netCDF_subset(parameters['export_netcdf'], [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
    items = [test_dict.extract_data()]
    items = np.array(items)
    print items.shape
    ds = Dataset_transformations(items,1000,items.shape)
    ds.twod_transformation()
    ds.normalize()
    clust_obj = utils.load_single(parameters['cluster_obj'])
    clust_obj.desc_date(export_template)
    cd = clust_obj.centroids_distance(ds,features_first=True)
    test_date = datetime.datetime.strptime(parameters['test_netcdf'].split('.')[0],'%Y-%m-%d_%H:%M:%S')
    test_date = datetime.datetime.strftime(test_date,'%y-%m-%d-%H')
    cluster_date = clust_obj._desc_date[cd[0][0]]
    cluster_date = datetime.datetime.strptime(cluster_date,'%Y-%m-%d_%H:%M:%S')
    cluster_date = datetime.datetime.strftime(cluster_date,'%y-%m-%d-%H')
    stat_dist = []
    for station in parameters['stations']:
        test_disp = Dataset(parameters['test_disp_path']+test_date+'/'+station['name']+'-'+test_date+'.nc','r')
        cluster_disp = Dataset(parameters['cluster_disp_path']+cluster_date+'/'+station['name']+'-'+cluster_date+'.nc','r')
        for pollutant in parameters['pollutants']:
            current_weather = test_disp.variables[pollutant['name']][:]
            cluster_weather = cluster_disp.variables[pollutant['name']][:]
            stat_dist.append((np.linalg.norm(current_weather-cluster_weather),pollutant['name'],station['name']))
    stat_dist = sorted(stat_dist, key=lambda x: x[0], reverse=False)
    print stat_dist
