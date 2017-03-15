import sys
sys.path.append('..')

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
import os
import dataset_utils as utils
import time


def reconstruct_date(date_str, dot_nc=False):
    if dot_nc:
        date = datetime.datetime.strptime(
            date_str.split('.')[0], '%Y-%m-%d_%H:%M:%S')
    else:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')
    return datetime.datetime.strftime(date, '%y-%m-%d-%H')

def calc_station_distance(parameters, test_date, cluster_date):
    stat_dist = []
    for station in parameters['stations']:
        jstation = {}
        jstation['name'] = station['name']
        polls = []
        test_disp = Dataset(parameters[
                            'test_disp_path'] + test_date + '/' + station['name']
                            + '-' + test_date + '.nc', 'r')
        cluster_disp = Dataset(parameters[
                               'cluster_disp_path'] + cluster_date + '/' + station['name']
                               + '-' + cluster_date + '.nc', 'r')
        for pollutant in parameters['pollutants']:
            jpollutant = {}
            jpollutant['name'] = pollutant['name']
            current_weather = test_disp.variables[pollutant['name']][:]
            cluster_weather = cluster_disp.variables[pollutant['name']][:]
            jpollutant['euclidean'] = unicode(np.linalg.norm(
                cluster_weather - current_weather))
            current_weather = current_weather.flatten()
            cluster_weather = cluster_weather.flatten()
            current_weather = np.divide(
                current_weather, np.sum(current_weather))
            cluster_weather = np.divide(
                cluster_weather, np.sum(cluster_weather))
            current_weather = np.add(current_weather, 1e-6)
            cluster_weather = np.add(cluster_weather, 1e-6)
            jpollutant['KL'] = unicode(scipy.stats.entropy(
                current_weather, cluster_weather))
            polls.append(jpollutant)
        jstation['distances'] = polls
        stat_dist.append(jstation)
    return stat_dist

def reorder_for_save(parameters,stat_dist):
    json_array = []
    poll_names = []
    for poll in parameters['pollutants']:
        poll_names.append(poll['name'])
    poll_euclidean = []
    poll_kl = []
    for pos,p in enumerate(poll_names):
        local_eucl = []
        local_kl = []
        for stat in stat_dist:
            s_name = stat['name']
            for dpos,dist in enumerate(stat['distances']):
                if dpos == pos:
                    local_eucl.append((s_name,dist['euclidean']))
                    local_kl.append((s_name,dist['KL']))
        poll_euclidean.append(local_eucl)
        poll_kl.append(local_kl)
    for pos,p in enumerate(poll_names):
        jpoll = {}
        jpoll['name'] = p
        jpoll['euclidean'] = poll_euclidean[pos]
        jpoll['KL'] = poll_kl[pos]
        json_array.append(jpoll)
    return json_array


def print_order(stat_dist):
    tupples = []
    for stat in stat_dist:
        for dist in stat['distances']:
            tupples.append((dist['euclidean'],dist['KL'],dist['name'],stat['name']))
    print sorted(tupples,key=lambda k: (k[0],k[1]),reverse=False)


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
    test_file_list = sorted(os.listdir(parameters['test_netcdf_path']))
    export_template = netCDF_subset(parameters['export_netcdf'], [700], [
                                    'GHT'], lvlname='num_metgrid_levels', timename='Times')
    clust_obj = utils.load_single(parameters['cluster_obj'])
    clust_obj.desc_date(export_template)
    for num,tfl in enumerate(test_file_list):
        start_time = time.time()
        print tfl
        print str((num+1))+'/'+str(len(test_file_list))
        test_dict = netCDF_subset(parameters['test_netcdf_path'] + tfl, [
                                  700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
        items = [test_dict.extract_data()]
        items = np.array(items)
        print items.shape
        ds = Dataset_transformations(items, 1000, items.shape)
        ds.twod_transformation()
        ds.normalize()
        cd = clust_obj.centroids_distance(ds, features_first=True)
        test_date = reconstruct_date(tfl, dot_nc=True)
        cluster_results = []
        try:
            os.chdir(parameters['test_disp_path']+test_date)
        except OSError:
            continue
        os.system('bzip2 -dk *.bz2')
        for i,cdi in enumerate(cd):
            jcluster = {}
            cluster_date = reconstruct_date(clust_obj._desc_date[cdi[0]])
            station_results = reorder_for_save(parameters,calc_station_distance(parameters, test_date, cluster_date))
            jcluster['name'] = clust_obj._desc_date[cdi[0]]
            jcluster['id'] = cdi[0]
            jcluster['distance'] = cdi[1]
            jcluster['results'] = station_results
            cluster_results.append(jcluster)
        cluster_results = sorted(cluster_results, key=lambda k: float(k['distance']),reverse=False)
        utils.save(parameters['export_path']+'/'+test_date+'.zip',cluster_results)
        os.system('find . ! -name \'*.bz2\' -type f -exec rm -f {} +')
        end_time = time.time()
        print ((start_time-end_time) / 60.0)
