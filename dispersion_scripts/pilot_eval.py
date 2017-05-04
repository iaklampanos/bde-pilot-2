import sys
sys.path.append('../..')

import os
import numpy as np
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
import csv
import datetime
import time
import ast
import json
from netcdf_subset import netCDF_subset
from Dataset_transformations import Dataset_transformations
from Detection import Detection
import dataset_utils as utils

DEF_VARS = ['GHT']
DEF_LEVELS = [700]
MODEL_FILE = ''
DISPERSION_PATH = '/mnt/disk1/thanasis/hysplit/hysplit/trunk/exec/raw_kmeans'
NETCDF_PATH = '/mnt/disk1/thanasis/data/wrf/nc/'


def load_clustering(clustering_path):
    clust_obj = utils.load_single(clustering_path)
    return clust_obj


def timing(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def load_weather_data(weather_id, netcdf_path=NETCDF_PATH, vars=DEF_VARS, levels=DEF_LEVELS):
    """ load weather from file corresponding to id """
    weather_id = datetime.datetime.strptime(weather_id,'%y-%m-%d-%H')
    weather_id = datetime.datetime.strftime(weather_id,'%Y-%m-%d_%H:%M:%S')
    test_dict = netCDF_subset(netcdf_path + weather_id + '.nc', levels, vars, lvlname='num_metgrid_levels', timename='Times')
    items = [test_dict.extract_data()]
    items = np.array(items)
    return items


def clean_weather(weather_data):
    ds = Dataset_transformations(weather_data, 1000, weather_data.shape)
    x = weather_data.shape[4]
    y = weather_data.shape[5]
    ds.twod_transformation()
    ds.normalize()
    return ds


def load_dispersions(cluster_date, dispersion_path=DISPERSION_PATH):
    disperions = []
    for station in sorted(os.listdir(dispersion_path + '/' + cluster_date)):
        if station.endswith('.nc'):
            nc_file = Dataset(dispersion_path + '/' +
                              cluster_date + '/' + station, 'r')
            disperions.append((station.split('-')[0], nc_file))
    return disperions


def rank_dispersions(dispersions, points, station_id,criteria):
    results = []
    for disp in dispersions:
        llat = []
        llon = []
        latlon = ast.literal_eval(points)
        for i in xrange(0, len(latlon), 3):
            llat.append(latlon[i])
            llon.append(latlon[i + 1])

        conc = np.sum(disp[1].variables['C137'][:, 0, :, :], axis=0)
        grid_lat = disp[1].variables['latitude'][:]
        grid_lon = disp[1].variables['longitude'][:]
        det_obj = Detection(conc, grid_lat,
                            grid_lon, llat, llon)
        det_obj.get_indices()
        det_obj.create_detection_map()
        if criteria == 'KL':
            results.append((disp[0], det_obj.KL()))
        else:
            results.append((disp[0], det_obj.cosine()))

        if criteria == 'KL':
            results = sorted(results, key=lambda k: k[1], reverse=False)
        else:
            results = sorted(results, key=lambda k: k[1], reverse=True)
    return results



if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-m', '--model', type=str,
                        help='neuran network object')
    parser.add_argument('-t', '--netcdf_path', type=str,
                        help='path of netcdf files')
    parser.add_argument('-c', '--clustering_object', type=str,
                        help='pickled clustering object path')
    parser.add_argument('-d', '--dispersions_path', type=str,
                        help='cluster dispersions path')
    parser.add_argument('-u', '--dispersion_criteria', type=str,
                        help='criteria of dispersion KL/cosine')
    opts = parser.parse_args()
    getter = attrgetter('model', 'netcdf_path', 'clustering_object',
                        'dispersions_path', 'dispersion_criteria')
    model_file,nc_path,clustering_path,disp_path,criteria = getter(opts)


    clust_obj = load_clustering(clustering_path)
    weather_id = ''
    for line in sys.stdin:
        line = line.strip()
        tokens = line.split(',')
        if tokens[0] != weather_id:
            weather_id = tokens[0]
            weather_data = load_weather_data(weather_id)
            cleaned_weather = clean_weather(weather_data)

            if MODEL_FILE != '':
                cleaned_weather.shift()
                cleaned_weather._items = exper._nnet.get_hidden(
                    np.transpose(cleaned_weather.get_items()))

        cd = clust_obj.centroids_distance(
            cleaned_weather, features_first=True)  # TODO alternative distances??

        cluster_date = utils.reconstruct_date(clust_obj._desc_date[cd[0][0]])

        # TODO adjust based on command line param
        dispersions = load_dispersions(cluster_date)

        results = rank_dispersions(dispersions, line.split('"')[1], tokens[1],criteria)

        top3 = results[:3]

        for pos, stat in enumerate(top3):
            if stat[0] == tokens[1]:
                eval_point = pos + 1
                break
            else:
                eval_point = 0
        sys.stdout.write(tokens[0]+','+tokens[1]+','+line.split('"')[1]+','+repr(top3)+','+str(eval_point)+'\n')
