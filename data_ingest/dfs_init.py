import requests
import psycopg2
import json
import getpass
import base64

# Create weather files directory
req = requests.put('http://namenode:50070/webhdfs/v1/sc5/weather?op=MKDIRS')
resp = req.json()
# Check if mkdir was successful
if resp['boolean'] == 'false':
    print '> Error on initializing directory '
    exit(-1)
# Create cluster dispersion files directory
req = requests.put('http://namenode:50070/webhdfs/v1/sc5/clusters?op=MKDIRS')
resp = req.json()
# Check if mkdir was successful
if resp['boolean'] == 'false':
    print '> Error on initializing directory '
    exit(-1)
# Create neural network model directory 
req = requests.put('http://namenode:50070/webhdfs/v1/sc5/models?op=MKDIRS')
resp = req.json()
# Check if mkdir was successful
if resp['boolean'] == 'false':
    print '> Error on initializing directory '
    exit(-1)
# Create class dispersion files directory
req = requests.put('http://namenode:50070/webhdfs/v1/sc5/classes?op=MKDIRS')
resp = req.json()
# Check if mkdir was successful
if resp['boolean'] == 'false':
    print '> Error on initializing directory '
    exit(-1)
# Get information about where the postgres sql db so we can get access to it
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
    # repeat until info are correct
    try:
        conn = psycopg2.connect("dbname='" + dnam + "' user='" + du +
                                "' host='" + dip + "' port='" + dpo + "'password='" + dpass + "'")
        break
    except:
        pass
# Save db info to json
dbobj = {}
dbobj['dbname'] = dnam
dbobj['host'] = dip
dbobj['port'] = dpo
dbobj['user'] = du
dbobj['pass'] = base64.b64encode(dpass)
with open('db_info.json', 'w') as outfile2:
    json.dump(dbobj, outfile2)
cur = conn.cursor()
# Create tables
cur.execute("CREATE TABLE weather (filename varchar(500),\
            hdfs_path varchar(2000),date timestamp,wind_dir500 json,wind_dir700 json,wind_dir900 json,GHT BYTEA, PRIMARY KEY(date))")
cur.execute("CREATE TABLE models (origin varchar(500),filename varchar(500),\
            hdfs_path varchar(2000),html varchar(2000), PRIMARY KEY(origin))")
cur.execute("CREATE TABLE cluster (filename varchar(500),\
            hdfs_path varchar(2000),station varchar(100),date timestamp,origin varchar(500),descriptor varchar(500),c137 json,i131 json,c137_pickle BYTEA,i131_pickle BYTEA, PRIMARY KEY(station,date,origin), FOREIGN KEY(origin) REFERENCES models(origin))")
cur.execute("CREATE TABLE class (filename varchar(500),\
            hdfs_path varchar(2000),station varchar(100),date timestamp,c137 json,i131 json,c137_pickle BYTEA,i131_pickle BYTEA, PRIMARY KEY(station,date))")
conn.commit()
cur.close()
conn.close()
