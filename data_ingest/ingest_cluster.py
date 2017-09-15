"""
   INFO
   -------------------------------------------------------------------------------------------
   Ingest cluster script uploads the netCDF dispersion files of each cluster descriptor of
   a certain clustering method.

   INPUT:
   -i, --input      : Directory that contains cluster descriptor sub-directories
   -m, --method     : Clustering method that produced said descriptors
   -d, --descriptor : Descriptor type (e.g km2,dense, etc.)
   -hp, --hdfs_path : Indicates where to upload the dispersion files in the Hdfs/Webhdfs

   Example run:
   python ingest_cluster.py -i <netcdf_dispersion_files_dir>/ -m '<clustering_method>' -d
   '<descriptor>' -hp <hdfs_path>
   -------------------------------------------------------------------------------------------
"""

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
    # Creates npy files representing the latitude and longitude ranges of the grid
    # Loop over input directories to find ANY netCDF dispersion file
    for lfl in sorted(os.listdir(inp)):
        for nc in sorted(os.listdir(inp + '/' + lfl)):
            if nc.endswith("nc"):
                # found netCDF file
                loc_path = inp + '/' + lfl + '/' + nc
                netcdf = Dataset(loc_path, 'r')
                lat = netcdf.variables['latitude'][:]
                lon = netcdf.variables['longitude'][:]
                np.save(inp + '/' + lfl + '/'+'lat.npy',lat)
                np.save(inp + '/' + lfl + '/'+'lon.npy',lon)
                # If not already uploaded to hdfs, then upload
                try:
                    hdfs.create_file('/sc5/clusters/lat.npy', open(inp + '/' + lfl + '/'+'lat.npy', 'rb'))
                    hdfs.create_file('/sc5/clusters/lon.npy', open(inp + '/' + lfl + '/'+'lon.npy', 'rb'))
                except:
                    pass
                # Exit loop and delete local files
                os.system('rm '+inp + '/' + lfl + '/'+'lat.npy')
                os.system('rm '+inp + '/' + lfl + '/'+'lon.npy')
                break
        break

if __name__ == '__main__':
    parser = ArgumentParser(description='Upload cluster descriptor disperion netcdf files')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input path')
    parser.add_argument('-m', '--method', required=True, type=str,
                        help='clustering method')
    parser.add_argument('-d', '--descriptor', required=True, type=str,
                        help='descriptor')
    parser.add_argument('-hp', '--hdfs_path', required=True, type=str,
                        help='hdfs path')
    opts = parser.parse_args()
    getter = attrgetter('input', 'method', 'hdfs_path', 'descriptor')
    inp, method, hp, desc = getter(opts)
    # Sort sub-directories by date
    local_filelist = sorted(os.listdir(inp))
    # Load db info
    with open('db_info.json', 'r') as data_file:
        dbpar = json.load(data_file)
    # Request db password
    dpass = getpass.getpass()
    # Connect to DB
    conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                            "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    cur = conn.cursor()
    # Init webhdfs client
    hdfs = PyWebHdfsClient(host='namenode', port='50070')
    make_latlon_files(inp,hp)
    # For each sub-directory
    for lfl in local_filelist:
        print lfl
        # For each file in sub-directory
        for nc in sorted(os.listdir(inp + '/' + lfl)):
            # Find netCDF dispersion files
            if nc.endswith("nc"):
               loc_path = inp + '/' + lfl + '/' + nc
               netcdf = Dataset(loc_path, 'r')
               print loc_path
               print hp + '/' + nc
               # Upload netCDF file to hdfs
               hdfs.create_file(hp + '/' + nc, open(loc_path, 'rb'))
               # Get dispersion as a single frame
               c137_pickle = np.sum(netcdf.variables['C137'][
                                    :, 0, :, :], axis=0)
               i131_pickle = np.sum(netcdf.variables['I131'][
                                    :, 0, :, :], axis=0)
               path = "http://namenode:50070/webhdfs/v1" + hp + "/" + nc + "?op=OPEN"
               date = datetime.datetime.strptime(lfl, '%y-%m-%d-%H')
               print date
               # Insert into db
               sql = "INSERT INTO cluster(filename,hdfs_path,station,date,origin,descriptor,c137,i131,c137_pickle,i131_pickle) VALUES (%s,%s,%s,TIMESTAMP %s,%s,%s,%s,%s,%s,%s)"
               cur.execute(sql, (nc, path, nc.split('-')[0],
               datetime.datetime.strftime(
                       date, '%m-%d-%Y %H:%M:%S'),
               method,desc, 'null', 'null', psycopg2.Binary(
                   cPickle.dumps(c137_pickle, 1)),
               psycopg2.Binary(cPickle.dumps(i131_pickle, 1))))
               netcdf.close()
    # Commit db changes
    conn.commit()
    cur.close()
    conn.close()
