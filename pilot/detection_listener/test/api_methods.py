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
import datetime
import time
from geojson import Feature, Point, MultiPoint, MultiLineString, LineString, FeatureCollection
import cPickle
import gzip
from sklearn.preprocessing import maxabs_scale, scale, minmax_scale
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import json
import threading
import Queue
import base64
import itertools
from dbconn import DBConn

APPS_ROOT = os.path.dirname(os.path.abspath(__file__))

conn = DBConn().engine

def dispersion_integral(dataset_name):
    dataset = Dataset(APPS_ROOT + '/' + dataset_name, 'r')
    dsout = Dataset(APPS_ROOT + '/' + 'int_' + dataset_name,
                    'w', format='NETCDF3_CLASSIC')
    c137 = dataset.variables['C137'][:]
    i131 = dataset.variables['I131'][:]
    c137 = np.sum(c137, axis=0).reshape(501, 501)
    i131 = np.sum(i131, axis=0).reshape(501, 501)
    for gattr in dataset.ncattrs():
        gvalue = dataset.getncattr(gattr)
        dsout.setncattr(gattr, gvalue)
    for dname, dim in dataset.dimensions.iteritems():
        if dname == 'time':
            dsout.createDimension(dname, 1 if not dim.isunlimited() else None)
        else:
            dsout.createDimension(dname, len(
                dim) if not dim.isunlimited() else None)
    print dsout.dimensions
    for v_name, varin in dataset.variables.iteritems():
        if v_name == 'C137':
            outVar = dsout.createVariable(
                v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k)
                              for k in varin.ncattrs()})
            outVar[:] = c137[:]
        elif v_name == 'I131':
            outVar = dsout.createVariable(
                v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k)
                              for k in varin.ncattrs()})
            outVar[:] = i131[:]
        else:
            try:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[:]
            except:
                outVar[:] = varin[0]
    dsout.close()


def calc_winddir(dataset_name, level):
    dataset = Dataset(APPS_ROOT + '/' + dataset_name, 'r')
    u = dataset.variables['UU'][:, level, :, range(0, 64)].reshape(13, 4096)
    v = dataset.variables['VV'][:, level, range(0, 64), :].reshape(13, 4096)
    lat = dataset.variables['XLAT_M'][0, :, :].flatten()
    lon = dataset.variables['XLONG_M'][0, :, :].flatten()
    u = np.sum(u, axis=0)
    v = np.sum(v, axis=0)
    uv = np.vstack((u, v))
    uv = np.divide(uv, np.max(uv))
    x1 = lon
    y1 = lat
    x1 = [float(i) for i in lon]
    y1 = [float(i) for i in lat]
    x2 = []
    y2 = []
    arr = []
    for i in range(0, uv.shape[1]):
        x2.append(float(x1[i] + uv[0][i]))
        y2.append(float(y1[i] + uv[1][i]))
    arr = []
    for i in range(0, uv.shape[1]):
        L1 = math.sqrt((x1[i] - x2[i]) * (x1[i] - x2[i]) +
                       (y2[i] - y1[i]) * (y2[i] - y1[i]))
        L2 = float(L1 / 3.5)
        x3 = x2[i] + (L2 / L1) * ((x1[i] - x2[i]) * math.cos((math.pi / 6)
                                                             ) + (y1[i] - y2[i]) * math.sin((math.pi) / 6))
        y3 = y2[i] + (L2 / L1) * ((y1[i] - y2[i]) * math.cos((math.pi / 6)
                                                             ) - (x1[i] - x2[i]) * math.sin((math.pi) / 6))
        x4 = x2[i] + (L2 / L1) * ((x1[i] - x2[i]) * math.cos((math.pi / 6)
                                                             ) - (y1[i] - y2[i]) * math.sin((math.pi) / 6))
        y4 = y2[i] + (L2 / L1) * ((y1[i] - y2[i]) * math.cos((math.pi / 6)
                                                             ) + (x1[i] - x2[i]) * math.sin((math.pi) / 6))
        a = (x1[i], y1[i])
        b = (x2[i], y2[i])
        c = (x3, y3)
        d = (x4, y4)
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


def load_class_weather(cur, date, origin):
    resp = cur.execute("select filename,hdfs_path,GHT,EXTRACT(EPOCH FROM TIMESTAMP '" +
                date + "' - date)/3600/24 as diff from weather order by diff desc;")
    row = resp.fetchone()
    if 'mult' in origin:
        items = cPickle.loads(str(row[2]))
        items = items.reshape(items.shape[0], -1)
        items = minmax_scale(items.sum(axis=0))
    else:
        items = cPickle.loads(str(row[2]))
        items = items[:, 1, :, :]
        items = minmax_scale(items.sum(axis=0))
    return items


def load_models(det_map, items, models, origin):
    for m in models:
        if origin == m[0]:
            if not('mult' in origin):
                items = items.reshape(1, 1, items.shape[0], items.shape[1])
                det_map = det_map.reshape(
                    1, 1, det_map.shape[0], det_map.shape[1])
                cl = m[1].get_output(items, det_map)[0].argsort()
                cl = list(cl)
                cl = [int(c) for c in cl if c < 18]
                cl = cl[:3]
            else:
                items = items.reshape(1, 1, 3, 64, 64)
                det_map = det_map.reshape(
                    1, 1, det_map.shape[0], det_map.shape[1])
                cl = m[1].get_output(items, det_map)[0].argsort()
                cl = list(cl)
                cl = [int(c) for c in cl if c < 18]
                cl = cl[:3]
    return cl


def worker(batch,q,pollutant,det_map):
    disp_results = []
    for row in batch:
        if pollutant == 'C137':
            det = cPickle.loads(str(row[2]))
        else:
            det = cPickle.loads(str(row[3]))
        det = scipy.misc.imresize(det, (167, 167))
        det = maxabs_scale(det)
        disp_results.append(
            (row[0], 1 - scipy.spatial.distance.cosine(det.flatten(), det_map.flatten())))
    q.put(disp_results)

def calc_scores(cur, items, cln, pollutant,det_map,origin):
    resp = cur.execute(
        "SELECT date,hdfs_path,c137_pickle,i131_pickle from class where station=\'" + cln + "\';")
    res = resp.fetchall()
    batch_size = len(res) / 4
    idx = xrange(0,len(res),batch_size)
    queue = Queue.Queue()
    disp_results = []
    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, args=(res[idx[i]:idx[i]+batch_size],queue,pollutant,det_map))
        threads.append(t)
        t.start()
        disp_results.append(queue.get())
    disp_results = list(itertools.chain.from_iterable(disp_results))
    print len(disp_results)
    disp_results = sorted(disp_results, key=lambda k: k[1], reverse=True)
    resp = cur.execute("SELECT date,GHT from weather;")
    res = resp.fetchall()
    weather_results = []
    for row in res:
        if 'mult' in origin:
            citems = cPickle.loads(str(row[1]))
            citems = citems.reshape(citems.shape[0], -1)
            citems = minmax_scale(citems.sum(axis=0))
        else:
            citems = cPickle.loads(str(row[1]))
            citems = citems[:, 1, :, :]
            citems = minmax_scale(citems.sum(axis=0))
        weather_results.append(
            (row[0], 1 - scipy.spatial.distance.cosine(items.flatten(), citems.flatten())))
    return disp_results, weather_results


def get_disp_frame(cur, cln, pollutant, results):
    dispersions = []
    scores = []
    print results[0]
    resp = cur.execute("select filename,hdfs_path,date,c137,i131 from class where  date=TIMESTAMP \'" +
                datetime.datetime.strftime(results[0], '%m-%d-%Y %H:%M:%S') + "\' and station='" + cln + "';")
    row = resp.fetchone()
    if (row[3] == None) or (row[4] == None):
        urllib.urlretrieve(row[1], row[0])
        dispersion_integral(row[0])
        os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                  row[0] + '\\":C137 ' + row[0].split('.')[0] + '_c137.tiff')
        os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                  row[0] + '\\":I131 ' + row[0].split('.')[0] + '_i131.tiff')
        os.system('make png TIFF_IN=' +
                  row[0].split('.')[0] + '_c137.tiff')
        os.system('make png TIFF_IN=' +
                  row[0].split('.')[0] + '_i131.tiff')
        os.system('make clean')
        with open(APPS_ROOT+ '/' + row[0].split('.')[0] + '_c137.json', 'r') as c137:
            c137_json = json.load(c137)
        with open(APPS_ROOT + '/' + row[0].split('.')[0] + '_i131.json', 'r') as i131:
            i131_json = json.load(i131)
        cur.execute("UPDATE class SET  c137=\'" +
                    json.dumps(c137_json) + "\' WHERE filename=\'" + row[0] + "\'")
        cur.execute("UPDATE class SET  i131=\'" +
                    json.dumps(i131_json) + "\' WHERE filename=\'" + row[0] + "\'")
        conn.commit()
        os.system('rm ' + APPS_ROOT + '/' +
                  row[0].split('.')[0] + '_c137.json')
        os.system('rm ' + APPS_ROOT + '/' +
                  row[0].split('.')[0] + '_i131.json')
        os.system('rm ' + APPS_ROOT + '/' + row[0])
        os.system('rm ' + APPS_ROOT + '/' + 'int_' + row[0])
        # os.system('rm ' + APPS_ROOT + '/' + res[0])
        if pollutant == 'C137':
            dispersion = json.dumps(c137_json)
        else:
            dispersion = json.dumps(i131_json)
        dispersions.append(dispersion)
        scores.append(round(results[1], 3))
    else:
        # os.system('rm ' + APPS_ROOT + '/' + res[0])
        if pollutant == 'C137':
            dispersion = json.dumps(row[3])
        else:
            dispersion = json.dumps(row[4])
        dispersions.append(dispersion)
        scores.append(round(results[1], 3))
    return scores, dispersions

def cdetections(cur, models, lat_lon, date, pollutant, metric, origin):
    items = load_class_weather(cur, date, origin)
    (filelat, filelon, llat, llon) = load_lat_lon(lat_lon)
    det_obj = Detection(np.zeros(shape=(501, 501)),
                        filelat, filelon, llat, llon)
    det_obj.get_indices()
    det_obj.create_detection_map(resize=True)
    det_map = det_obj._det_map
    cl = load_models(det_map, items, models, origin)
    resp = cur.execute("SELECT station from class group by station order by station;")
    res = resp.fetchall()
    res = [i for i in res]
    print res
    print cl
    class_name = [str(res[i][0]) for i in cl]
    print class_name
    import time
    time.sleep(60)
    for cln in class_name:
        (disp_results, weather_results) = calc_scores(cur, items, cln, pollutant, det_map, origin)
        for w in weather_results:
            if w[0] == disp_results[0][0]:
                d = disp_results[0]
                results = (d[0],w[1]*d[1])
        try:
            scores, dispersions = get_disp_frame(cur, cln, pollutant, results)
        except:
            d = disp_results[0]
            results = (d[0],d[1])
            scores, dispersions = get_disp_frame(cur, cln, pollutant, results)
    scores, dispersions, class_name = zip(
        *sorted(zip(scores, dispersions, class_name), key=lambda k: k[0], reverse=True))
    print scores
    send = {}
    send['stations'] = class_name
    send['scores'] = scores
    send['dispersions'] = dispersions
    return json.dumps(send)


def load_lat_lon(lat_lon):
    llat = []
    llon = []
    for llobj in lat_lon:
        llat.append(float(llobj['lat']))
        llon.append(float(llobj['lon']))
    urllib.urlretrieve(
        'http://namenode:50070/webhdfs/v1/sc5/clusters/lat.npy?op=OPEN', 'lat.npy')
    urllib.urlretrieve(
        'http://namenode:50070/webhdfs/v1/sc5/clusters/lon.npy?op=OPEN', 'lon.npy')
    filelat = np.load('lat.npy')
    filelon = np.load('lon.npy')
    os.system('rm ' + APPS_ROOT + '/' + 'lat.npy')
    os.system('rm ' + APPS_ROOT + '/' + 'lon.npy')
    return filelat, filelon, llat, llon


def load_weather_data(cur, date, origin):
    resp = cur.execute("select filename,hdfs_path,GHT,EXTRACT(EPOCH FROM TIMESTAMP '" +
                date + "' - date)/3600/24 as diff from weather order by diff desc;")
    res = resp.fetchone()
    if 'mult' in origin:
        items = cPickle.loads(str(res[2]))
        items = items.reshape(1, 1, items.shape[0], items.shape[
                              1], items.shape[2], items.shape[3])
    else:
        items = cPickle.loads(str(res[2]))
        items = items[:, 1, :, :]
        items = items.reshape(
            1, 1, items.shape[0], 1, items.shape[1], items.shape[2])
    return items,res


def load_cluster_date(items, models, origin):
    ds = Dataset_transformations(items, 1000, items.shape)
    x = items.shape[2]
    y = items.shape[3]
    ds.twod_transformation()
    ds.normalize()
    for m in models:
        if origin == m[0]:
            if 'kmeans' not in m[0]:
                clust_obj = m[2]
                ds._items = m[1].get_hidden(ds._items.T)
                cd = clust_obj.centroids_distance(ds, features_first=False)
                cluster_date = utils.reconstruct_date(
                    clust_obj._desc_date[cd[0][0]])
            else:
                clust_obj = m[1]
                cd = clust_obj.centroids_distance(ds, features_first=True)
                cluster_date = utils.reconstruct_date(
                    clust_obj._desc_date[cd[0][0]])
    return cluster_date


def calc_station_scores(cur, lat_lon, timestamp, origin, descriptor, pollutant):
    (filelat, filelon, llat, llon) = load_lat_lon(lat_lon)
    results = []
    res = cur.execute("select filename,hdfs_path,station,c137_pickle,i131_pickle from cluster where date=TIMESTAMP \'" +
                datetime.datetime.strftime(timestamp, '%m-%d-%Y %H:%M:%S') + "\' and origin='" + origin + "' and descriptor='" + descriptor + "'")
    for row in res:
        if pollutant == 'C137':
            det_obj = Detection(cPickle.loads(
                str(row[3])), filelat, filelon, llat, llon)
            det_obj.get_indices()
            det_obj.create_detection_map()
        else:
            det_obj = Detection(cPickle.loads(
                str(row[4])), filelat, filelon, llat, llon)
            det_obj.get_indices()
            det_obj.create_detection_map()
        if det_obj.calc() != 0:
            results.append((row[2], det_obj.cosine()))
        else:
            results.append((row[2], 0))
    return results


def get_top3_stations(cur, top3, timestamp, origin, pollutant):
    top3_names = [top[0] for top in top3]
    top3_scores = [round(top[1], 3) for top in top3]
    stations = []
    scores = []
    dispersions = []
    resp = cur.execute("select filename,hdfs_path,station,c137,i131 from cluster where date=TIMESTAMP \'" +
                datetime.datetime.strftime(timestamp, '%m-%d-%Y %H:%M:%S') + "\' and origin='" + origin + "'")
    rows = resp.fetchall()
    for row in rows:
        if row[2] in top3_names:
            if (row[3] == None) or (row[4] == None):
                urllib.urlretrieve(row[1], row[0])
                dispersion_integral(row[0])
                os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                          row[0] + '\\":C137 ' + row[0].split('.')[0] + '_c137.tiff')
                os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                          row[0] + '\\":I131 ' + row[0].split('.')[0] + '_i131.tiff')
                os.system('make png TIFF_IN=' +
                          row[0].split('.')[0] + '_c137.tiff')
                os.system('make png TIFF_IN=' +
                          row[0].split('.')[0] + '_i131.tiff')
                os.system('make clean')
                with open(row[0].split('.')[0] + '_c137.json', 'r') as c137:
                    c137_json = json.load(c137)
                with open(row[0].split('.')[0] + '_i131.json', 'r') as i131:
                    i131_json = json.load(i131)
                cur.execute("UPDATE cluster SET  c137=\'" +
                            json.dumps(c137_json) + "\' WHERE filename=\'" + row[0] + "\'")
                cur.execute("UPDATE cluster SET  i131=\'" +
                            json.dumps(i131_json) + "\' WHERE filename=\'" + row[0] + "\'")
                conn.commit()
                os.system('rm ' + APPS_ROOT + '/' +
                          row[0].split('.')[0] + '_c137.json')
                os.system('rm ' + APPS_ROOT + '/' +
                          row[0].split('.')[0] + '_i131.json')
                os.system('rm ' + APPS_ROOT + '/' + row[0])
                os.system('rm ' + APPS_ROOT + '/' + 'int_' + row[0])
                # os.system('rm ' + APPS_ROOT + '/' + res[0])
                stations.append(str(row[2]))
                scores.append(top3_scores[top3_names.index(row[2])])
                if pollutant == 'C137':
                    dispersions.append(json.dumps(c137_json))
                else:
                    dispersions.append(json.dumps(i131_json))
            else:
                # os.system('rm ' + APPS_ROOT + '/' + res[0])
                stations.append(str(row[2]))
                scores.append(top3_scores[top3_names.index(row[2])])
                if pollutant == 'C137':
                    dispersions.append(json.dumps(row[3]))
                else:
                    dispersions.append(json.dumps(row[4]))
    return scores, dispersions, stations


def detections(cur, models, lat_lon, date, pollutant, metric, origin):
    (items,res) = load_weather_data(cur, date, origin)
    cluster_date = load_cluster_date(items, models, origin)
    descriptor = origin.split('_')
    descriptor = descriptor[len(descriptor) - 1]
    timestamp = datetime.datetime.strptime(cluster_date, '%y-%m-%d-%H')
    results = calc_station_scores(cur, lat_lon, timestamp, origin, descriptor, pollutant)
    results = sorted(results, key=lambda k: k[1] if k[
                     1] > 0 else float('inf'), reverse=False)
    top3 = results[:3]
    print top3
    scores, dispersions, stations = get_top3_stations(cur, top3, timestamp, origin, pollutant)
    scores, dispersions, stations = zip(
        *sorted(zip(scores, dispersions, stations), key=lambda k: k[0] if k[0] > 0 else float('inf'), reverse=False))
    send = {}
    send['stations'] = stations
    send['scores'] = scores
    send['dispersions'] = dispersions
    return json.dumps(send)


def get_methods(cur):
    res = cur.execute("select origin,html from models;")
    origins = []
    for row in res:
        origin = {}
        origin['html'] = row[1]
        origin['origin'] = row[0]
        origins.append(origin)
    return json.dumps(origins)


def get_closest(cur, date, level):
    level = int(level)
    if level == 22:
        resp = cur.execute("select filename,hdfs_path,wind_dir500,EXTRACT(EPOCH FROM TIMESTAMP '" +
                    date + "' - date)/3600/24 as diff from weather group by date\
                    having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    elif level == 26:
        resp = cur.execute("select filename,hdfs_path,wind_dir700,EXTRACT(EPOCH FROM TIMESTAMP '" +
                    date + "' - date)/3600/24 as diff from weather group by date\
                    having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    elif level == 33:
        resp = cur.execute("select filename,hdfs_path,wind_dir900,EXTRACT(EPOCH FROM TIMESTAMP '" +
                    date + "' - date)/3600/24 as diff from weather group by date\
                    having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    res = resp.fetchone()
    if res[3] > 5:
        return json.dumps({'error': 'date is out of bounds'})
    if res[2] == None:
        urllib.urlretrieve(res[1], res[0])
        json_dir = calc_winddir(res[0], level)
        os.system('rm ' + APPS_ROOT + '/' + res[0])
        if level == 22:
            cur.execute("UPDATE weather SET  wind_dir500=\'" +
                        json_dir + "\' WHERE filename=\'" + res[0] + "\'")
            conn.commit()
        elif level == 26:
            cur.execute("UPDATE weather SET  wind_dir700=\'" +
                        json_dir + "\' WHERE filename=\'" + res[0] + "\'")
            conn.commit()
        elif level == 33:
            cur.execute("UPDATE weather SET  wind_dir900=\'" +
                        json_dir + "\' WHERE filename=\'" + res[0] + "\'")
            conn.commit()
        return json_dir
    else:
        return json.dumps(res[2])
