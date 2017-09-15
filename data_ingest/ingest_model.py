"""
   INFO
   -------------------------------------------------------------------------------------------
   Ingest model script uploads the respective model of each estimation method.

   INPUT:
   -i, --input  : Absolute path of model_template zip
   -m, --method : Estimation method (e.g shallow_ae_km2)
   -ht, --html  : String that presents the estimation method to the end user
                  (e.g Shallow autoencoder clustering (km2) )

   Example run:
   python ingest_model.py -i /pilot_data/<model_template.zip> -m
   "<clustering/classification method>" -ht "<html_repr>"
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
import gzip,cPickle


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='model path')
    parser.add_argument('-m', '--method', required=True, type=str,
                        help='clustering method')
    parser.add_argument('-ht', '--html', required=True, type=str,
                        help='html description')
    opts = parser.parse_args()
    getter = attrgetter('input','method','html')
    inp,method,html = getter(opts)
    # Get list of uploaded models
    req = requests.get('http://namenode:50070/webhdfs/v1/sc5/models?op=LISTSTATUS')
    resp = req.json()
    fl = resp['FileStatuses']['FileStatus']
    hdfs_list = []
    for file in fl:
        hdfs_list.append(file['pathSuffix'])
    # Load db info
    with open('db_info.json','r') as data_file:
        dbpar = json.load(data_file)
    # Request db password
    dpass = getpass.getpass()
    # Connect to db
    conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                            "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    cur = conn.cursor()
    # Init webhdfs client
    hdfs = PyWebHdfsClient(host='namenode', port='50070')
    # If model is not already uploaded, then upload
    if inp not in hdfs_list:
        print inp
        hpath = inp.split('/')
        hpath = hpath[len(hpath)-1]
        # Upload to hdfs
        hdfs.create_file('/sc5/models/'+hpath, open(inp,'rb'))
        path = "http://namenode:50070/webhdfs/v1/sc5/models/"+hpath+"?op=OPEN"
        # Insert to db
        cur.execute("INSERT INTO models(origin,filename,hdfs_path,html) VALUES(\'"+method+"\',\'"+hpath+"\',\'"+path+"\',\'"+html+"\')")
    # Commit changes
    conn.commit()
    cur.close()
    conn.close()
