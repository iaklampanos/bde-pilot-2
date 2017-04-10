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
    opts = parser.parse_args()
    getter = attrgetter('input')
    inp = getter(opts)
    local_filelist = sorted(os.listdir(inp))
    req = requests.get('http://namenode:50070/webhdfs/v1/sc5/weather?op=LISTSTATUS')
    resp = req.json()
    fl = resp['FileStatuses']['FileStatus']
    hdfs_list = []
    for file in fl:
        hdfs_list.append(file['pathSuffix'])
    intersect = list(set(local_filelist).intersection(hdfs_list))
    with open('db_info.json','r') as data_file:
        dbpar = json.load(data_file)
    dpass = getpass.getpass()
    conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                            "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    cur = conn.cursor()
    hdfs = PyWebHdfsClient(host='namenode', port='50070')
    for lfl in local_filelist:
        if lfl not in intersect:
            print lfl
            hdfs.create_file('/sc5/weather/'+lfl, open(inp+lfl,'rb'))
            # req = requests.put('http://namenode:50070/webhdfs/v1/sc5/weather/'+lfl+'?op=CREATE',allow_redirects=False)
            # location = req.headers['location']
            # print location
            # file_data={'file': open(inp+lfl, 'r')}
            # req_upl = requests.put(location,files=file_data, headers={'content-type': 'application/octet-stream'})
            path = "http://namenode:50070/webhdfs/v1/sc5/weather/"+lfl+"?op=OPEN"
            date = datetime.datetime.strptime(lfl.split('.')[0],'%Y-%m-%d_%H-%M-%S')
            cur.execute("INSERT INTO weather(filename,hdfs_path,date,wind_dir) VALUES(\'"+lfl+"\',\'"+path+"\', TIMESTAMP \'"+datetime.datetime.strftime(date,'%m-%d-%Y %H:%M:%S')+"\',null)")
    conn.commit()
    cur.close()
    conn.close()
