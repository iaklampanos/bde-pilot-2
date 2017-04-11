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
    # req = requests.get('http://namenode:50070/webhdfs/v1'+hp+'?op=LISTATUS')
    # resp = req.json()
    # fl = resp['FileStatuses']['FileStatus']
    # hdfs_list = []
    # for file in fl:
    #     hdfs_list.append(file['pathSuffix'])

    with open('db_info.json','r') as data_file:
        dbpar = json.load(data_file)
    dpass = getpass.getpass()
    conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                            "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    cur = conn.cursor()
    hdfs = PyWebHdfsClient(host='namenode', port='50070')
    for lfl in local_filelist:
        # intersect = list(set(os.listdir(inp+'/'+lfl)).intersection(hdfs_list))
        for nc in sorted(os.listdir(inp+'/'+lfl)):
            if nc.endswith("nc"):
            #    if nc not in intersect:
               loc_path = inp+'/'+lfl+'/'+nc
               print loc_path
               print hp+'/'+nc
               hdfs.create_file(hp+'/'+nc, open(loc_path,'rb'))
               path = "http://namenode:50070/webhdfs/v1"+hp+"/"+nc+"?op=OPEN"
               date = datetime.datetime.strptime(lfl,'%y-%m-%d-%H')
               print date
               cur.execute("INSERT INTO cluster(filename,hdfs_path,station,date,origin,c137,i131) \
                VALUES(\'"+nc+"\',\'"+path+"\',\'"+nc.split('-')[0]+"\',TIMESTAMP \'"+datetime.datetime.strftime(date,'%m-%d-%Y %H:%M:%S')+"\',\'"+method+"\',null,null)")
    conn.commit()
    cur.close()
    conn.close()
