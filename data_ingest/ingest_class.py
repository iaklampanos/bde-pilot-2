"""
   INFO
   -------------------------------------------------------------------------------------------
   Ingest class script uploads the netCDF dispersion files of each test dataset 3 day range.

   INPUT:
   sys.argv[1:] : Directories containing said netCDF dispersion files.

   Example run:
   ls -d /pilot_data/<netcdf_dispersion_files_dir>/* >> nc.txt
   cat nc.txt | xargs  xargs -n <# of files per processor> -P <# of processors> python
   ingest_class.py
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
import base64
import sys

# Load DB Info
with open('db_info.json','r') as data_file:
    dbpar = json.load(data_file)
# Connect to DB
conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                        "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + base64.b64decode(dbpar['pass']) + "'")
cur = conn.cursor()
# Init webhdfs client
hdfs = PyWebHdfsClient(host='namenode', port='50070')

def main():
    # hadoop path
    hp = '/sc5/classes'
    # for each argument usually given with xargs for parallel ingestion,
    # arguments should be directories containing the dispersions of a date for
    # each station.
    for lfl in sys.argv[1:]:
        for nc in sorted(os.listdir(lfl)):
            # Find netCDF files
            if nc.endswith("nc"):
                loc_path = lfl + '/' + nc
                netcdf = Dataset(loc_path, 'r')
                print loc_path
                print hp + '/' + nc
                # upload to hdfs
                hdfs.create_file(hp + '/' + nc, open(loc_path, 'rb'))
                # get dispersion as a single frame
                c137_pickle = np.sum(netcdf.variables['C137'][
                                     :, 0, :, :], axis=0)
                i131_pickle = np.sum(netcdf.variables['I131'][
                                     :, 0, :, :], axis=0)
                path = "http://namenode:50070/webhdfs/v1" + hp + "/" + nc + "?op=OPEN"
                # reformat date
                dstr = lfl.split('/')
                dstr = dstr[len(dstr)-1]
                date = datetime.datetime.strptime(dstr, '%y-%m-%d-%H')
                # insert into db
                sql = "INSERT INTO class(filename,hdfs_path,station,date,c137,i131,c137_pickle ,i131_pickle) VALUES (%s,%s,%s,TIMESTAMP %s,%s,%s,%s,%s)"
                cur.execute(sql, (nc, path, nc.split('-')[0],
                datetime.datetime.strftime(
                        date, '%m-%d-%Y %H:%M:%S'),
                'null', 'null', psycopg2.Binary(
                    cPickle.dumps(c137_pickle, 1)),
                psycopg2.Binary(cPickle.dumps(i131_pickle, 1))))
                netcdf.close()
    # Commit changes to DB
    conn.commit()
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
