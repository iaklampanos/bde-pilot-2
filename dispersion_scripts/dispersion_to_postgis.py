import sys
sys.path.append('..')

import psycopg2
from operator import attrgetter
from argparse import ArgumentParser
from netCDF4 import Dataset
import numpy as np
import dataset_utils as utils
import json
from netcdf_subset import netCDF_subset
import datetime
import os
import time
from PreparingCursor import PreparingCursor

def drop_schema(parameters):
    db = psycopg2.connect('host=' + parameters['db_host'] +
                          ' dbname=' + parameters['db_name'] +
                          ' user=' + parameters['db_username'] +
                          ' password=' + parameters['db_password'])
    cur = db.cursor()
    cur.execute('DROP TABLE dispersions;')
    db.commit()
    cur.close()
    db.close()



def create_schema(parameters):
    db = psycopg2.connect('host=' + parameters['db_host'] +
                          ' dbname=' + parameters['db_name'] +
                          ' user=' + parameters['db_username'] +
                          ' password=' + parameters['db_password'])
    cur = db.cursor()
    cur.execute(
        "CREATE TABLE dispersions (gid serial PRIMARY KEY not null,conc float,date varchar(500),\
        pollutant varchar(30),station varchar(100),xyz varchar(30),type varchar(100),\
        location GEOGRAPHY(POINT,4326));")
    db.commit()
    cur.close()
    db.close()

from scipy.io import netcdf

def populate(parameters, path, date, type='CLUSTER'):
    db = psycopg2.connect('host=' + parameters['db_host'] +
                          ' dbname=' + parameters['db_name'] +
                          ' user=' + parameters['db_username'] +
                          ' password=' + parameters['db_password'])
    cur = db.cursor(cursor_factory=PreparingCursor)
    for station in parameters['stations']:
        stat_st = time.time()
        # disp_nc = Dataset(path + '/' + station['name']
        #                        + '-' + date + '.nc', 'r')
        disp_nc = netcdf.netcdf_file(path + '/' + station['name']
                               + '-' + date + '.nc', 'r')
        times = disp_nc.variables['time'].data.tolist()
        lat = disp_nc.variables['latitude'].data.tolist()
        lon = disp_nc.variables['longitude'].data.tolist()
        for pid, pollutant in enumerate(parameters['pollutants']):
            for pos, t in enumerate(times):
                # start = time.time()
                values = []
                for la, _lat in enumerate(lat):
                    for lo, _lon in enumerate(lon):
                        cdisp = float(disp_nc.variables[
                            pollutant['name']].data[pos, :, la, lo])
                        xyz = str(pos) + '_' + str(la) + '_' + str(lo)
                        values.append("({},\'{}\',\'{}\',\'{}\',\'{}\',\'{}\',ST_MakePoint({},{}))".format(str(cdisp),str(date),str(pollutant['name']),str(station['name']),str(xyz),str(type),str(_lat),str(_lon)))
                str_values = ""
                for num, v in enumerate(values):
                    if num != (len(values) - 1):
                        str_values += v + ','
                    else:
                        str_values += v
                cur.execute("INSERT INTO dispersions (conc,date,pollutant,station,xyz,type,location) VALUES "+str_values+";")
                # end = time.time()
                # hours, rem = divmod(end-start, 3600)
                # minutes, seconds = divmod(rem, 60)
                # print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                db.commit()
    db.commit()
    cur.close()
    db.close()


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
    # create_schema(parameters)
    # drop_schema(parameters)
    test_file_list = sorted(os.listdir(parameters['path']))
    for num, tfl in enumerate(test_file_list):
        path = parameters['path'] + '/' + tfl
        os.chdir(path)
        os.system('bzip2 -dk *.bz2')
        st = time.time()
        populate(parameters, path, tfl)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        os.system('find . ! -name \'*.bz2\' -type f -exec rm -f {} +')
