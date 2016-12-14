from operator import attrgetter
from argparse import ArgumentParser
import os

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-sd', '--start_date', required=True, type=str,
                        help='output path')
    parser.add_argument('-ed', '--end_date', required=True, type=str,
                        help='output path')
    opts = parser.parse_args()
    getter = attrgetter('input', 'start_date', 'end_date')
    inp, start, end = getter(opts)
    #start_parts = start.split('_')
    #end_parts = end.split('_')
    os.system('csh /home/wrf/Build_WRF/LIBRARIES/WPS/link_grib.csh '+
                                                       '/home/wrf/data/grib/')
    nwps = open('/home/wrf/data/namelist.wps', 'r')
    nwps_new = open('/home/wrf/Build_WRF/LIBRARIES/WPS/namelist.wps', 'w')
    for i, line in enumerate(nwps.readlines()):
        if i == 3:
            nwps_new.write(' start_date = \'' + start + '\'\n')
            continue
        elif i == 4:
            nwps_new.write(' end_date = \'' + end + '\'\n')
        else:
            nwps_new.write(line)
    nwps.close()
    nwps_new.close()
    os.system('/home/wrf/Build_WRF/LIBRARIES/WPS/./ungrib.exe')
    os.system('/home/wrf/Build_WRF/LIBRARIES/WPS/./metgrid.exe')
    os.system('rm /home/wrf/Build_WRF/LIBRARIES/WPS/FILE*')
    os.system('rm /home/wrf/Build_WRF/LIBRARIES/WPS/GRIBFILE*')
    os.system('mv /home/wrf/Build_WRF/LIBRARIES/WPS/met_em*'+
                                   ' /home/wrf/Build_WRF/LIBRARIES/WRFV3/run/')
