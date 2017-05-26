import sys
sys.path.append('..')

import datetime
import os

DESCRIPTOR_PATH='/mnt/disk1/thanasis/NIPS/clusters/double_kmeans/GHT_700_raw_kmeans/descriptors/'
GRIB_3D = ''
GRIB_2D = ''
OUTPUT_PATH = ''
INVARIANT = ''
HYSPLIT_PATH = ''

def reconstruct_date(date_str, dot_nc=False):
    if dot_nc:
        date = datetime.datetime.strptime(
            date_str.split('.')[0], '%Y-%m-%d_%H:%M:%S')
    else:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')
    return datetime.datetime.strftime(date, '%Y-%m-%dT%H:%M:%S')

def date_range(date_str):
    start = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
    date_range = [start+datetime.timedelta(hours=x) for x in xrange(0,78,6)]
    return [datetime.datetime.strftime(date, '%Y-%m-%dT%H:%M:%S') for date in date_range]

def reconstruct_range(date_list,prefix=None):
    if prefix is None:
        date_list = str(date_list)
        date_list = date_list.replace('[','')
        date_list = date_list.replace(']','')
        date_list = date_list.replace('\'','')
        date_list = date_list.replace(',','')
    else:
        date_list = [prefix + '/' + date for date in date_list]
        date_list = str(date_list)
        date_list = date_list.replace('[','')
        date_list = date_list.replace(']','')
        date_list = date_list.replace('\'','')
        date_list = date_list.replace(',','')
    return date_list

def main():
    for descriptor_nc in sys.argv[1:]:
        gb_str = reconstruct_date(descriptor_nc,dot_nc=True)
        date_list = date_range(gb_str)
        full_2d = OUTPUT_PATH+'/'+gb_str+'_2d.grib'
        full_3d = OUTPUT_PATH+'/'+gb_str+'_3d.grib'
        os.system('cdo merge'+reconstruct_range(date_list,prefix=GRIB_3D)+' '+full_3d
        os.system('cdo merge'+reconstruct_range(date_list,prefix=GRIB_2D)' '+full_2d
        os.chdir(HYSPLIT_PATH)
        os.system('make 3D_IN='+full_3d+' 2D_IN='+full_2d+' INVARIANT='+INVARIANT+' BIN_NAME='+gb_str+' convert_f conc_f clean')

if __name__ == "__main__":
    main()
