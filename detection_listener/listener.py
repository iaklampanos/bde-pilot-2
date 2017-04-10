import sys
sys.path.append('..')

import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'

from web import app
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
from netcdf_subset import netCDF_subset
from Dataset_transformations import Dataset_transformations
from Detection import Detection
import dataset_utils as utils
import numpy as np
from netCDF4 import Dataset
import urllib
import psycopg2
import getpass
import os
import math
from geojson import Feature,Point,MultiPoint,MultiLineString,LineString,FeatureCollection

BOOTSTRAP_SERVE_LOCAL = True
app = Flask(__name__)
CORS(app)

app.config.from_object(__name__)


inp = None
parameters = None
export_template = None
clust_obj = None
exper = None
conn = None
cur = None
# dpass = getpass.getpass()
dpass = 'postgres'
APPS_ROOT = os.path.dirname(os.path.abspath(__file__))

def calc_winddir(dataset_name):
    dataset = Dataset(APPS_ROOT+'/'+dataset_name, 'r')
    u = dataset.variables['UU'][:,27,:,range(0,64)].reshape(13,4096)
    v = dataset.variables['VV'][:,27,range(0,64),:].reshape(13,4096)
    lat = dataset.variables['XLAT_M'][0,:,:].flatten()
    lon = dataset.variables['XLONG_M'][0,:,:].flatten()
    u = np.sum(u,axis=0)
    v = np.sum(v,axis=0)
    uv = np.vstack((u,v))
    uv = np.divide(uv,np.max(uv))
    x1 = lon
    y1 = lat
    x1 = [float(i) for i in lon]
    y1 = [float(i) for i in lat]
    x2 = []
    y2 = []
    arr = []
    for i in range(0,uv.shape[1]):
        x2.append(float(x1[i]+uv[0][i]))
        y2.append(float(y1[i]+uv[1][i]))
    arr = []
    for i in range(0,uv.shape[1]):
        L1 = math.sqrt((x1[i]-x2[i])*(x1[i]-x2[i])+(y2[i]-y1[i])*(y2[i]-y1[i]))
        L2 = float(L1 / 3.5)
        x3 = x2[i] + (L2/L1)*((x1[i]-x2[i])*math.cos((math.pi/6))+(y1[i]-y2[i])*math.sin((math.pi)/6))
        y3 = y2[i] + (L2/L1)*((y1[i]-y2[i])*math.cos((math.pi/6))-(x1[i]-x2[i])*math.sin((math.pi)/6))
        x4 = x2[i] + (L2/L1)*((x1[i]-x2[i])*math.cos((math.pi/6))-(y1[i]-y2[i])*math.sin((math.pi)/6))
        y4 = y2[i] + (L2/L1)*((y1[i]-y2[i])*math.cos((math.pi/6))+(x1[i]-x2[i])*math.sin((math.pi)/6))
        a = (x1[i],y1[i])
        b = (x2[i],y2[i])
        c = (x3,y3)
        d = (x4,y4)
        temp = []
        temp.append(a)
        temp.append(b)
        temp2 = []
        temp2.append(b)
        temp2.append(c)
        temp3 = []
        temp3.append(b)
        temp3.append(d)
        arr.append(temp)
        arr.append(temp2)
        arr.append(temp3)
    feature = Feature(geometry=MultiLineString(arr))
    dataset.close()
    return json.dumps(feature)



@app.route('/detections/<file_name>/<pollutant>', methods=['POST'])
def detections(file_name, pollutant):
    lat_lon = request.get_json(force=True)
    llat = []
    llon = []
    for llobj in lat_lon:
        llat.append(float(llobj['lat']))
        llon.append(float(llobj['lon']))
    test_dict = netCDF_subset(parameters['test_netcdf_path'] + file_name, [
        parameters['level']], [parameters['var']], lvlname='num_metgrid_levels', timename='Times')
    items = [test_dict.extract_data()]
    items = np.array(items)
    ds = Dataset_transformations(items, 1000, items.shape)
    x = items.shape[4]
    y = items.shape[5]
    ds.twod_transformation()
    ds.normalize()
    try:
        if parameters['autoenc'] == 'simple':
            ds.shift()
            ds._items = exper._nnet.get_hidden(np.transpose(ds.get_items()))
            print ds._items.shape
            cd = clust_obj.centroids_distance(ds, features_first=False)
        elif parameters['autoenc'] == 'conv':
            ds.shift()
            items = ds.get_items()
            items = items.reshape(items.shape[1],1,x,y)
            print items.shape
            ds._items = exper._nnet.get_hidden(items)
            ds._items = ds._items.astype(np.float32)
            cd = clust_obj.centroids_distance(ds, features_first=False)
    except:
        cd = clust_obj.centroids_distance(ds, features_first=True)
    cluster_date = utils.reconstruct_date(clust_obj._desc_date[cd[0][0]])
    path = parameters['dispersion_path'] + '/' + cluster_date
    results = []
    results2 = []
    for station in parameters['stations']:
        nc_file = Dataset(
            path + '/' + station['name'] + '-' + cluster_date + '.nc', 'r')
        filelat = nc_file.variables['latitude'][:]
        filelon = nc_file.variables['longitude'][:]
        det_obj = Detection(nc_file, pollutant, filelat, filelon, llat, llon)
        det_obj.get_indices()
        det_obj.calculate_concetration()
        det_obj.create_detection_map()
        results.append((station['name'], det_obj.calc()))
        # results.append((station['name'], det_obj.calc_score()))
        # results2.append((station['name'], det_obj.calc_score2()))
    # print sorted(results2, key=lambda k: k[1], reverse=False)
    # results = sorted(results, key=lambda k: k[1], reverse=False)
    results = sorted(results, key=lambda k: k[1], reverse=True)
    print results
    send = {}
    send["station"] = str(results[0][0])
    send["date"] = str(utils.reconstruct_date(clust_obj._desc_date[cd[0][0]]))
    send["pollutant"] = str(pollutant)
    send["score"] = str(results[0][1])
    return json.dumps(send)

@app.route('/getClosest/<date>', methods=['GET'])
def get_closest(date):
    cur.execute("select filename,hdfs_path,wind_dir,EXTRACT(EPOCH FROM TIMESTAMP '"+date+"' - date)/3600/24 as diff from weather order by diff desc;");
    res = cur.fetchone()
    print res
    if res[2] == None:
        urllib.urlretrieve(res[1],res[0])
        json_dir = calc_winddir(res[0])
        os.system('rm '+APPS_ROOT+'/'+res[0])
        cur.execute("UPDATE weather SET  wind_dir=\'"+json_dir+"\' WHERE filename=\'"+res[0]+"\'")
        conn.commit()
        return json.dumps(json_dir)
    else:
        return json.dumps(res[2])

if __name__ == '__main__':
    with open('db_info.json','r') as data_file:
        dbpar = json.load(data_file)
    conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                            "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    cur = conn.cursor()
    inp = 'parameters.json'
    with open(inp) as data_file:
        parameters = json.load(data_file)
    export_template = netCDF_subset(parameters['export_netcdf'],  [parameters['level']], [
                                    parameters['var']], lvlname='num_metgrid_levels', timename='Times')
    clust_obj = utils.load_single(parameters['cluster_obj'])
    clust_obj.desc_date(export_template)
    try:
        exper = utils.load_single(parameters['autoenc_model'])
    except:
        pass
    try:
        app.run(host='0.0.0.0', debug=True)
    except Exception:
        pass
