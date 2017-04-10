import json
import requests
from operator import attrgetter
from argparse import ArgumentParser
import os
import psycopg2
import datetime
import getpass
from pywebhdfs.webhdfs import PyWebHdfsClient

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input path')
    parser.add_argument('-m', '--method', required=True, type=str,
                        help='clustering method')
    parser.add_argument('-hp', '--hdfs_path', required=True, type=str,
                        help='hdfs path')
    opts = parser.parse_args()
    getter = attrgetter('input','method','hdfs_path')
    inp,method,hp = getter(opts)
    local_filelist = sorted(os.listdir(inp))
    # req = requests.get('http://namenode:50070/webhdfs/v1/sc5/clusters/'+hp+'?op=LISTATUS')
    # fl = resp['FileStatuses']['FileStatus']
    # hdfs_list = []
    # for file in fl:
    #     hdfs_list.append(file['pathSuffix'])
    # intersect = list(set(local_filelist).intersection(hdfs_list))
    # with open('db_info.json','r') as data_file:
    #     dbpar = json.load(data_file)
    # dpass = getpass.getpass()
    # conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
    #                         "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    # cur = conn.cursor()
    # hdfs = PyWebHdfsClient(host='namenode', port='50070')
    for lfl in local_filelist:
        print lfl
        for nc in sorted(os.listdir(lfl)):
            if nc.endswith('.nc'):
                print nc

            # if lfl not in intersect:
            # print lfl
            # hdfs.create_file('/sc5/weather/'+lfl, open(inp+lfl,'rb'))
            # path = "http://namenode:50070/webhdfs/v1/sc5/weather/"+lfl+"?op=OPEN"
            # date = datetime.datetime.strptime(lfl.split('.')[0],'%Y-%m-%d_%H-%M-%S')
            # cur.execute("INSERT INTO weather(filename,hdfs_path,date,wind_dir) VALUES(\'"+lfl+"\',\'"+path+"\', TIMESTAMP \'"+datetime.datetime.strftime(date,'%m-%d-%Y %H:%M:%S')+"\',null)")
    # conn.commit()
    # cur.close()
    # conn.close()
