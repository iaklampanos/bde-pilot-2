import sys
sys.path.append('..')

import datetime
import os
import time

DESCRIPTOR_PATH='/mnt/disk1/thanasis/NIPS/clusters/double_kmeans/GHT_700_raw_kmeans/descriptors/'
GRIB_3D = '/mnt/share300/data/ECMWF/3d/hxygd6'
GRIB_2D = '/mnt/share300/data/ECMWF/2d'
OUTPUT_PATH = '/mnt/share300/thanasis/descriptors'
INVARIANT = '/mnt/share300/data/ECMWF/invariant.grib'
HYSPLIT_PATH = '/mnt/share300/thanasis/hysplit/'

def reconstruct_date(date_str, dot_nc=False):
    if dot_nc:
        date = datetime.datetime.strptime(
            date_str.split('.')[0], '%Y-%m-%d_%H:%M:%S')
    else:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')
    return datetime.datetime.strftime(date, '%Y-%m-%dT%H:%M:%S')

def timing(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def date_range(date_str):
    start = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
    date_range = [start+datetime.timedelta(hours=x) for x in xrange(0,78,6)]
    return [datetime.datetime.strftime(date, '%Y-%m-%dT%H:%M:%S') for date in date_range]

def reconstruct_range(date_list,flag_2d=True,prefix=None):
    if prefix is None:
        if flag_2d:
           date_list = [date + '.grb' for date in date_list]
        else:
           date_list = [date + '.grib' for date in date_list]
        date_list = str(date_list)
        date_list = date_list.replace('[','')
        date_list = date_list.replace(']','')
        date_list = date_list.replace('\'','')
        date_list = date_list.replace(',','')
    else:
        if flag_2d:
           date_list = [prefix + '/' + date + '.grb' for date in date_list]
        else:
           date_list = [prefix + '/' + date + '.grib' for date in date_list]
        date_list = str(date_list)
        date_list = date_list.replace('[','')
        date_list = date_list.replace(']','')
        date_list = date_list.replace('\'','')
        date_list = date_list.replace(',','')
    return date_list

def main():
    for descriptor_nc in sys.argv[1:]:
        start = time.time()
        gb_str = reconstruct_date(descriptor_nc,dot_nc=True)
        date_list = date_range(gb_str)
        full_2d = OUTPUT_PATH+'/'+gb_str+'_2d.grib'
        full_3d = OUTPUT_PATH+'/'+gb_str+'_3d.grib'
        os.system('cdo merge '+reconstruct_range(date_list,flag_2d=False,prefix=GRIB_3D)+' '+full_3d)
        os.system('cdo merge '+reconstruct_range(date_list,prefix=GRIB_2D)+' '+full_2d)
        os.chdir(HYSPLIT_PATH)
        os.system('make 3D_IN='+full_3d+' 2D_IN='+full_2d+' INVARIANT='+INVARIANT+' BIN_NAME='+gb_str+' convert_f conc_f clean')
        timing(start,time.time())

if __name__ == "__main__":
    main()
