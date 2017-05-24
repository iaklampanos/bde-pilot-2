import os
import numpy as np
from operator import attrgetter
from argparse import ArgumentParser
import json
from netCDF4 import Dataset
import time
import csv

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input path')
    parser.add_argument('-n', '--point_num', required=True, type=int,
                        help='number of points')
    parser.add_argument('-rn', '--repeat_num', required=True, type=int,
                        help='number of loops')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='output path')
    opts = parser.parse_args()
    getter = attrgetter('input', 'point_num', 'repeat_num', 'output')
    inp, num, rnum, output = getter(opts)
    start = time.time()
    for test_id in sorted(os.listdir(inp)):
        print test_id
        f = open(output + '/' + test_id + '_' +
                 str(num) + '_points.csv', 'wt')
        writer = csv.writer(f)
        os.chdir(inp + '/' + test_id)
        os.system('bzip2 -dk *.bz2')
        for station in sorted(os.listdir(inp + '/' + test_id)):
            if station.endswith('.nc') and 'IGNALINA' not in station and 'GARONA' not in station:
                nc_stat = Dataset(station, 'r')
                c137 = np.sum(nc_stat.variables['C137'][:, 0, :, :], axis=0)
                lat = nc_stat.variables['latitude'][:]
                lon = nc_stat.variables['longitude'][:]
                nonzero = np.nonzero(c137)
                nonzero_points = np.array(
                    [(nonzero[0][i], nonzero[1][i]) for i in range(0, len(nonzero[0]))])
                inx = np.array(range(0, len(nonzero[0])))
                for loop in range(0,rnum):
                    randp = np.random.choice(inx, num)
                    points = []
                    latlonval = []
                    for i in randp:
                        point = {}
                        point['lat'] = float(lat[tuple(nonzero_points[i])[0]])
                        point['lon'] = float(lon[tuple(nonzero_points[i])[1]])
                        point['val'] = float(c137[tuple(nonzero_points[i])])
                        loc_lat = float(lat[tuple(nonzero_points[i])[0]])
                        loc_lon = float(lon[tuple(nonzero_points[i])[1]])
                        loc_val = float(c137[tuple(nonzero_points[i])])
                        latlonval.append(loc_lat)
                        latlonval.append(loc_lon)
                        latlonval.append(loc_val)
                        points.append(point)
                    writer.writerow((test_id, station.split('-')[0], latlonval))
        os.system('find . ! -name \'*.bz2\' -type f -exec rm -f {} +')
        f.close()
