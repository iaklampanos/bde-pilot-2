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
from sklearn.preprocessing import minmax_scale,maxabs_scale

COBJ_PATH='/mnt/disk1/thanasis/NIPS/clusters/density_based/GHT_700_conv_density.zip'
CLUSTER_DISP='/mnt/disk1/thanasis/NIPS/clusters/density_based/ght700_conv/dispersions/'
MODEL = '/mnt/disk1/thanasis/NIPS/models/conv_ght_700/CONV_GHT_700_model_cpu.zip'


TEST_PATH='/mnt/disk1/thanasis/data/wrf/nc/'
TEST_DISP='/mnt/disk1/thanasis/hysplit/hysplit/trunk/test_exec/test/'
STATIONS = ["CERNAVODA","KOZLODUY","GROHNDE","EMSLAND","SIZEWELL","HINKLEY","COFRENTES","ALMARAZ","LOVIISA","FORSMARK","RINGHALS","SUKRAINE","PAKS","KRSKO","VANDELLOS","DOEL","HEYSHAM","IGNALINA","GARONA"]
MODEL_FLAG = True


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
    cos_diffs = []
    for stat in STATIONS:
        npy_test = TEST_DISP+test_date+'/'+stat+'-'+test_date+'-c137.npy'
        npy_cluster = CLUSTER_DISP+cluster_date+'/'+stat+'-'+cluster_date+'-c137.npy'
        current_weather = maxabs_scale(np.load(npy_test))
        cluster_weather = maxabs_scale(np.load(npy_cluster))
        cos_diffs.append(scipy.spatial.distance.cosine(current_weather.flatten(),cluster_weather.flatten()))
    return cos_diffs


def main():
    clust_obj = utils.load_single(COBJ_PATH)
    if MODEL_FLAG:
        model = utils.load_single(MODEL)
    for test_nc in sys.argv[1:]:
        try:
            file = open(test_nc+'.csv','w')
            test_dict = netCDF_subset(TEST_PATH + test_nc, [
                                      700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
            items = [test_dict.extract_data()]
            items = np.array(items)
            ds = Dataset_transformations(items, 1000, items.shape)
            ds.twod_transformation()
            # ds.normalize()
            ds._items = minmax_scale(ds._items).T
            if MODEL_FLAG:
                items = ds.get_items()
                items = model.get_hidden(items)
                ds_hidden = Dataset_transformations(items,1000,items.shape)
                print ds_hidden._items.shape
                # utils.plot_pixel_image(items[4,:],model.get_output(items)[4,:],64,64)
                cd = clust_obj.centroids_distance(ds_hidden, features_first=False)
            else:
                cd = clust_obj.centroids_distance(ds, features_first=False)
            cluster_date = reconstruct_date(clust_obj._desc_date[cd[0][0]])
            test_date = reconstruct_date(test_nc, dot_nc=True)
            cos = get_results(test_date, cluster_date)
            for i,stat in enumerate(STATIONS):
                file.write(str(test_date)+','+str(cluster_date)+','+str(stat)+','+unicode(cos[i])+'\n')
        except Exception,e: print str(e)

if __name__ == "__main__":
    main()
