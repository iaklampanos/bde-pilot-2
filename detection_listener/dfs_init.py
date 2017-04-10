import requests
import psycopg2
import json
import getpass


req = requests.put('http://namenode:50070/webhdfs/v1/sc5/weather?op=MKDIRS')
resp = req.json()
if resp['boolean'] == 'false':
    print '> Error on initializing directory '
    exit(-1)
req = requests.put('http://namenode:50070/webhdfs/v1/sc5/clusters?op=MKDIRS')
resp = req.json()
if resp['boolean'] == 'false':
    print '> Error on initializing directory '
    exit(-1)
while True:
    print 'Database IP: '
    dip = raw_input()
    print 'Database port: '
    dpo = raw_input()
    print 'Database name: '
    dnam = raw_input()
    print 'Database user name : '
    du = raw_input()
    dpass = getpass.getpass()
    try:
        conn = psycopg2.connect("dbname='" + dnam + "' user='" + du +
                                "' host='" + dip + "' port='" + dpo + "'password='" + dpass + "'")
        break
    except:
        pass
dbobj = {}
dbobj['dbname'] = dnam
dbobj['host'] = dip
dbobj['port'] = dpo
dbobj['user'] = du
with open('db_info.json', 'w') as outfile2:
    json.dump(dbobj, outfile2)
cur = conn.cursor()
cur.execute("CREATE TABLE weather (filename varchar(500),\
            hdfs_path varchar(2000),date timestamp,wind_dir json, PRIMARY KEY(date))")
cur.execute("CREATE TABLE cluster (filename varchar(500),\
            hdfs_path varchar(2000),station varchar(100),date timestamp,origin varchar(500),disp_vis json,c137 json,i131 json, PRIMARY KEY(station,date,origin))")
conn.commit()
cur.close()
conn.close()
