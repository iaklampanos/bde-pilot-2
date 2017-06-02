import json
import requests
from operator import attrgetter
from argparse import ArgumentParser
import os
import psycopg2
import datetime
import getpass
from pywebhdfs.webhdfs import PyWebHdfsClient
from netCDF4 import Dataset
import numpy as np
import cPickle

def make_latlon_files(inp,hp):
    for lfl in sorted(os.listdir(inp)):
        for nc in sorted(os.listdir(inp + '/' + lfl)):
            if nc.endswith("nc"):
                loc_path = inp + '/' + lfl + '/' + nc
                netcdf = Dataset(loc_path, 'r')
                lat = netcdf.variables['latitude'][:]
                lon = netcdf.variables['longitude'][:]
                np.save(inp + '/' + lfl + '/'+'lat.npy',lat)
                np.save(inp + '/' + lfl + '/'+'lon.npy',lon)
                try:
                    hdfs.create_file('/sc5/classes/lat.npy', open(inp + '/' + lfl + '/'+'lat.npy', 'rb'))
                    hdfs.create_file('/sc5/classes/lon.npy', open(inp + '/' + lfl + '/'+'lon.npy', 'rb'))
                except:
                    pass
                os.system('rm '+inp + '/' + lfl + '/'+'lat.npy')
                os.system('rm '+inp + '/' + lfl + '/'+'lon.npy')
                break
        break

with open('db_info.json','r') as data_file:
    dbpar = json.load(data_file)
conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                        "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + base64.b64decode(dbpar['pass']) + "'")
cur = conn.cursor()
hdfs = PyWebHdfsClient(host='namenode', port='50070')

def main():
    for lfl in sys.argv[1:]:
        for nc in sorted(os.listdir(lfl)):
            if nc.endswith("nc"):
                loc_path = lfl + '/' + nc
                netcdf = Dataset(loc_path, 'r')
                print loc_path
                print hp + '/' + nc
                hdfs.create_file(hp + '/' + nc, open(loc_path, 'rb'))
                c137_pickle = np.sum(netcdf.variables['C137'][
                                     :, 0, :, :], axis=0)
                i131_pickle = np.sum(netcdf.variables['I131'][
                                     :, 0, :, :], axis=0)
                path = "http://namenode:50070/webhdfs/v1" + hp + "/" + nc + "?op=OPEN"
                date = datetime.datetime.strptime(lfl, '%y-%m-%d-%H')
                print date
                sql = "INSERT INTO class(filename,hdfs_path,station,date,c137,i131,c137_pickle ,i131_pickle) VALUES (%s,%s,%s,TIMESTAMP %s,%s,%s,%s,%s)"
                cur.execute(sql, (nc, path, nc.split('-')[0],
                datetime.datetime.strftime(
                        date, '%m-%d-%Y %H:%M:%S'),
                'null', 'null', psycopg2.Binary(
                    cPickle.dumps(c137_pickle, 1)),
                psycopg2.Binary(cPickle.dumps(i131_pickle, 1))))
                netcdf.close()
     conn.commit()
     cur.close()
     conn.close()


if __name__ == '__main__':
    main()
