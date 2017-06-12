"""
   INFO
   -------------------------------------------------------------------------------------------
   Ingest weather script uploads the unprocessed netCDF files containing 3 days worth of various
   weather variables and data.

   INPUT:
   sys.argv[1:] : netCDF files.

   Example run:
   ls -d <netcdf_weather_files_dir>/* >> weather.txt
   cat weather.txt | xargs -n <# of files per processor> -P <# of processors> python
   ingest_weather.py
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
from netcdf_subset import netCDF_subset
import sys
import base64
import cPickle

# Load db info
with open('db_info.json','r') as data_file:
    dbpar = json.load(data_file)
# Connect to DB
conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                        "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + base64.b64decode(dbpar['pass']) + "'")
cur = conn.cursor()
# Init webhdfs client
hdfs = PyWebHdfsClient(host='namenode', port='50070')

def main():
    # Get list of uploaded weather files
    req = requests.get('http://namenode:50070/webhdfs/v1/sc5/weather?op=LISTSTATUS')
    resp = req.json()
    fl = resp['FileStatuses']['FileStatus']
    # Loop for each file
    for f in sys.argv[1:]:
        # If file is not uploaded, then upload
        if not(f in fl):
            lfl = f.split('/')
            lfl = lfl[len(lfl)-1]
            print lfl
            # Get GHT values on 3 levels
            lfl_nc = netCDF_subset(f,[500,700,900],['GHT'])
            items = lfl_nc.extract_data()
            ght_pkl = items.reshape(items.shape[1:])
            # Upload netCDF file
            hdfs.create_file('/sc5/weather/'+lfl, open(f,'rb'))
            path = "http://namenode:50070/webhdfs/v1/sc5/weather/"+lfl+"?op=OPEN"
            # Reformat date
            date = datetime.datetime.strptime(lfl.split('.')[0],'%Y-%m-%d_%H-%M-%S')
            # Insert record to db
            sql = "INSERT INTO weather(filename,hdfs_path,date,wind_dir500,wind_dir700,wind_dir900,GHT) \
            VALUES(%s,%s,TIMESTAMP %s,%s,%s,%s,%s)"
            cur.execute(sql, (lfl,path,datetime.datetime.strftime(date,'%m-%d-%Y %H:%M:%S'),'null','null','null',psycopg2.Binary(
                cPickle.dumps(ght_pkl, 1))))
    # Commit changes
    conn.commit()
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
