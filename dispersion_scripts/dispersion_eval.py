#!/usr/bin/env python

import sys
sys.path.append('..')

from netcdf_subset import netCDF_subset
from Dataset_transformations import Dataset_transformations
import dataset_utils as utils
import os
import numpy as np
import datetime
import scipy
import time

TEST_PATH='/mnt/disk1/thanasis/data/wrf/nc/'
COBJ_PATH='/mnt/disk1/thanasis/NIPS/clusters/double_kmeans/GHT_700_deep.zip'
TEST_DISP='/mnt/disk1/thanasis/hysplit/hysplit/trunk/test_exec/test/'
CLUSTER_DISP='/mnt/disk1/thanasis/NIPS/clusters/double_kmeans/ght_700_deep/dispersions/'
STATIONS = ["CERNAVODA","KOZLODUY","GROHNDE","EMSLAND","SIZEWELL","HINKLEY","COFRENTES","ALMARAZ","LOVIISA","FORSMARK","RINGHALS","SUKRAINE","PAKS","KRSKO","VANDELLOS","DOEL","HEYSHAM","IGNALINA","GARONA"]

def reconstruct_date(date_str, dot_nc=False):
    if dot_nc:
        date = datetime.datetime.strptime(
            date_str.split('.')[0], '%Y-%m-%d_%H:%M:%S')
    else:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')
    return datetime.datetime.strftime(date, '%y-%m-%d-%H')

def timing(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def get_results(test_date, cluster_date):
    mse_diffs = []
    kl_diffs = []
    for stat in STATIONS:
        npy_test = TEST_DISP+test_date+'/'+stat+'-'+test_date+'-c137.npy'
        npy_cluster = CLUSTER_DISP+cluster_date+'/'+stat+'-'+cluster_date+'-c137.npy'
        current_weather = np.load(npy_test)
        cluster_weather = np.load(npy_cluster)
        MSE =  np.mean((current_weather - cluster_weather)**2)
        current_weather = current_weather.flatten()
        cluster_weather = cluster_weather.flatten()
        current_weather = np.divide(
            current_weather, np.sum(current_weather))
        cluster_weather = np.divide(
            cluster_weather, np.sum(cluster_weather))
        current_weather = np.add(current_weather, 1e-6)
        cluster_weather = np.add(cluster_weather, 1e-6)
        KL = scipy.stats.entropy(
            current_weather, cluster_weather)
        mse_diffs.append(MSE)
        kl_diffs.append(KL)
    return [mse_diffs,kl_diffs]


def main():
    clust_obj = utils.load_single(COBJ_PATH)
    model = utils.load_single('/mnt/disk1/thanasis/NIPS/models/GHT_700_deep/GHT_700_deep_model_cpu.zip')
    for test_nc in sys.argv[1:]:
        try:
            file = open(test_nc+'.csv','w')
            test_dict = netCDF_subset(TEST_PATH + test_nc, [
                                      700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
            items = [test_dict.extract_data()]
            items = np.array(items)
            ds = Dataset_transformations(items, 1000, items.shape)
            ds.twod_transformation()
            ds.normalize()
            items = np.transpose(ds._items)
            items = model.get_hidden(items)
            ds_hidden = Dataset_transformations(items,1000,items.shape)
            # utils.plot_pixel_image(items[4,:],model.get_output(items)[4,:],64,64)
            cd = clust_obj.centroids_distance(ds_hidden, features_first=False)
            cluster_date = reconstruct_date(clust_obj._desc_date[cd[0][0]])
            test_date = reconstruct_date(test_nc, dot_nc=True)
            [MSE,KL] = get_results(test_date, cluster_date)
            for i,stat in enumerate(STATIONS):
                file.write(str(test_date)+','+str(cluster_date)+','+str(stat)+','+unicode(MSE[i])+','+unicode(KL[i])+'\n')
        except Exception,e: print str(e)

if __name__ == "__main__":
    main()
