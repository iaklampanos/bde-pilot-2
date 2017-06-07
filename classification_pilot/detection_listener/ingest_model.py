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
    req = requests.get('http://namenode:50070/webhdfs/v1/sc5/models?op=LISTSTATUS')
    resp = req.json()
    fl = resp['FileStatuses']['FileStatus']
    hdfs_list = []
    for file in fl:
        hdfs_list.append(file['pathSuffix'])
    with open('db_info.json','r') as data_file:
        dbpar = json.load(data_file)
    dpass = getpass.getpass()
    conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                            "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    cur = conn.cursor()
    hdfs = PyWebHdfsClient(host='namenode', port='50070')
    if inp not in hdfs_list:
        print inp
        hdfs.create_file('/sc5/models/'+inp, open(inp,'rb'))
        path = "http://namenode:50070/webhdfs/v1/sc5/models/"+inp+"?op=OPEN"
        cur.execute("INSERT INTO models(origin,filename,hdfs_path,html) VALUES(\'"+method+"\',\'"+inp+"\',\'"+path+"\',\'"+html+"\')")
    conn.commit()
    cur.close()
    conn.close()
