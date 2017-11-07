"""
   CLASS INFO
   ---------------------------------------------------------------------------
   This class acts as the model (from MVC framework) for the SC5 #3 pilot.
   ---------------------------------------------------------------------------
"""

from Dataset_transformations import Dataset_transformations
from Detection import Detection
import dataset_utils as utils
import numpy as np
from netCDF4 import Dataset
import urllib
import getpass
import os
import math
import datetime
import time
from geojson import Feature, Point, MultiPoint, MultiLineString, LineString, FeatureCollection
import cPickle
from sklearn.preprocessing import maxabs_scale, scale, minmax_scale
from shapely.geometry import shape, Point, Polygon, mapping, MultiPolygon, MultiPoint
import scipy.misc
import json
import threading
import Queue
import itertools
from dbconn import DBConn
from itertools import chain

APPS_ROOT = os.path.dirname(os.path.abspath(__file__))

conn = DBConn().engine
semagrow_batch_size = 100

# TODO: Update function with gdal python API

# This Function returns a dispersion that consist of 72 hours as a single frame.
# For this purpose we calculate the integral of a dispersion.
def dispersion_integral(dataset_name):
    # Load NetCDF file from local path
    dataset = Dataset(APPS_ROOT + '/' + dataset_name, 'r')
    dsout = Dataset(APPS_ROOT + '/' + 'int_' + dataset_name,
                    'w', format='NETCDF3_CLASSIC')
    # Retrieve both pollutants
    c137 = dataset.variables['C137'][:]
    i131 = dataset.variables['I131'][:]
    # Calculate their sum
    c137 = np.sum(c137, axis=0).reshape(501, 501)
    i131 = np.sum(i131, axis=0).reshape(501, 501)
    # Write the disperion integrals on disk in NetCDF format.
    # We need the dispersion integrals in NetCDF format due to the fact that we
    # use integrals for visualization. In order to visualize geographical information
    # in our application we use a set of tools accesible through the OS, like
    # gdal_translate

    # Copy attributes from original file
    for gattr in dataset.ncattrs():
        gvalue = dataset.getncattr(gattr)
        dsout.setncattr(gattr, gvalue)
    # Copy dimensions from original file
    for dname, dim in dataset.dimensions.iteritems():
        if dname == 'time':
            dsout.createDimension(dname, 1 if not dim.isunlimited() else None)
        else:
            dsout.createDimension(dname, len(
                dim) if not dim.isunlimited() else None)
    print dsout.dimensions
    # Copy every other variable from the original file except from
    # the pollutant variables
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
            # Catch exception on time variable
            except:
                outVar[:] = varin[0]
    # Finish writing process
    dsout.close()

# This function calculates the wind direction from a NetCDF file.
# The wind direction is calculated and visualized by computing the dot product
# of the U wind direction and the V wind direction. After calculating the dot product
# we create arrows that represent the wind direction for the whole grid.
def calc_winddir(dataset_name, level):
    # Load NetCDF file from local file
    dataset = Dataset(APPS_ROOT + '/' + dataset_name, 'r')
    # Retrieve U Wind direction
    u = dataset.variables['UU'][:, level, :, range(0, 64)].reshape(13, 4096)
    # Retrieve V Wind direction
    v = dataset.variables['VV'][:, level, range(0, 64), :].reshape(13, 4096)
    # Retrieve Latitude and Longitude values
    lat = dataset.variables['XLAT_M'][0, :, :].flatten()
    lon = dataset.variables['XLONG_M'][0, :, :].flatten()
    # Calculate sum of all time frames for U and V wind direction
    u = np.sum(u, axis=0)
    v = np.sum(v, axis=0)
    # Turn UV into [0,1] vector
    uv = np.vstack((u, v))
    uv = np.divide(uv, np.max(uv))
    # Create 2 Points for each (lat,lon) pair in the grid
    # Point1 with coordinates of (latN,lonN)
    # Point2 with coordinates of (latN+UV[N],lonN+UV[N])
    x1 = lon
    y1 = lat
    x1 = [float(i) for i in lon]
    y1 = [float(i) for i in lat]
    x2 = []
    y2 = []
    # Calculate Point2
    for i in range(0, uv.shape[1]):
        x2.append(float(x1[i] + uv[0][i]))
        y2.append(float(y1[i] + uv[1][i]))
    # Placeholder for every pair of points
    arr = []
    # Calculate arrow between Point1 and Point2 basd on
    # https://math.stackexchange.com/questions/1314006/drawing-an-arrow
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
        # Add each point to placeholder array
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

# This function retrieves the closest weather file to a date given by the user.
# The weather is retrieved from POSTGRES in the form of pickled object.
# Weather files are used to exctact the GHT variable that was used in our
# experiments.
def load_class_weather(cur, date, origin):
    # Safe query is used due to multiple workers accessing the same database
    # and connection.
    resp = DBConn().safequery("select filename,hdfs_path,GHT,EXTRACT(EPOCH FROM TIMESTAMP '" +
                date + "' - date)/3600/24 as diff from weather group by date\
                having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    row = resp.fetchone()
    # If model has mult in its title then we need multiple levels (500,700,900 hPa)
    # of the GHT variable.
    if 'mult' in origin:
        # Load weather as a pickled object and scale the variable
        items = cPickle.loads(str(row[2]))
        items = items.reshape(items.shape[0], -1)
        items = minmax_scale(items.sum(axis=0))
    else:
        items = cPickle.loads(str(row[2]))
        # GHT 700hPa
        items = items[:, 1, :, :]
        items = minmax_scale(items.sum(axis=0))
    return items

# This function returns top3 predictions for classification models, it needs
# to be an individual function due to classfication models receiving both
# disperions(in the form of detection maps) and weather data as input.
def get_cprediction(det_map, items, models, origin):
    # Loop through every ingested model
    for m in models:
        # Get selected model
        if origin == m[0]:
            # Check if model accepts multiple levels of GHT var
            if not('mult' in origin):
                # Reshape dispersion and weather data in accepted form
                items = items.reshape(1, 1, items.shape[0], items.shape[1])
                det_map = det_map.reshape(
                    1, 1, det_map.shape[0], det_map.shape[1])
                # Get predictions
                cl = m[1].get_output(items, det_map)[0].argsort()
                cl = list(cl)
                cl = [int(c) for c in cl if c < 18]
                # Return top3 stations
                cl = cl[:3]
            else:
                # Reshape dispersion and weather data in accepted form
                items = items.reshape(1, 1, 3, 64, 64)
                det_map = det_map.reshape(
                    1, 1, det_map.shape[0], det_map.shape[1])
                # Get predictions
                cl = m[1].get_output(items, det_map)[0].argsort()
                cl = list(cl)
                cl = [int(c) for c in cl if c < 18]
                # Return top3 stations
                cl = cl[:3]
    return cl

# This function is used for parallel processing of every ingested dispersion
# in order to find the closest representation for visualization purposes.
def worker(batch,q,pollutant,det_map):
    disp_results = []
    # For each dispersion in given batch
    for row in batch:
        # Load dispersion
        if pollutant == 'C137':
            det = cPickle.loads(str(row[2]))
        else:
            det = cPickle.loads(str(row[3]))
        # Preprocessing of dispersion
        det = scipy.misc.imresize(det, (167, 167))
        det_shape = det.shape
        det = maxabs_scale(det.flatten(),axis=1)
        det = det.reshape(det_shape)
        # Get distance between real dispersion and detection points
        disp_results.append(
            (row[0], 1 - scipy.spatial.distance.cosine(det.flatten(), det_map.flatten())))
    # Return batch distances
    q.put(disp_results)

# This function calculates distance between the top3 predicted station disperions
# and the detection points (for visualization). This is necessary because when we
# treat the source estimation problem as classification there are no exported
# byproducts as clusters and therefore there is no clear candidate for dispersion
# visualization.
def calc_scores(cur, items, cln, pollutant,det_map,origin):
    # Get all dispersions for the predicted stations
    resp = DBConn().safequery(
        "SELECT date,hdfs_path,c137_pickle,i131_pickle from class where station=\'" + cln + "\';")
    res = resp.fetchall()
    # Calculate distances in 4 batches
    batch_size = len(res) / 4
    idx = xrange(0,len(res),batch_size)
    queue = Queue.Queue()
    disp_results = []
    threads = []
    # Split dispersions into 4 Threads
    for i in range(4):
        t = threading.Thread(target=worker, args=(res[idx[i]:idx[i]+batch_size],queue,pollutant,det_map))
        threads.append(t)
        t.start()
        disp_results.append(queue.get())
    disp_results = list(itertools.chain.from_iterable(disp_results))
    print len(disp_results)
    disp_results = sorted(disp_results, key=lambda k: k[1], reverse=True)
    # Having all the distances from the detection points and the real dispersions,
    # we need to select the dispersion that not only intersects with the detection
    # points but also the dispersion's origin weather is close to the real weather.
    resp = DBConn().safequery("SELECT date,GHT from weather;")
    res = resp.fetchall()
    weather_results = []
    # For each weather in given date
    for row in res:
        # Check if model expects multiple levels of GHT var
        if 'mult' in origin:
            citems = cPickle.loads(str(row[1]))
            citems = citems.reshape(citems.shape[0], -1)
            citems = minmax_scale(citems.sum(axis=0))
        else:
            citems = cPickle.loads(str(row[1]))
            citems = citems[:, 1, :, :]
            citems = minmax_scale(citems.sum(axis=0))
        # Get distance
        weather_results.append(
            (row[0], 1 - scipy.spatial.distance.cosine(items.flatten(), citems.flatten())))
    # Return all distances
    return disp_results, weather_results

# This function returns the estimated station dispersion in the GeoJSON form which
# is expected from the visualization mechanism.
def get_disp_frame(cur, cln, pollutant, results):
    dispersions = []
    scores = []
    print results[0]
    # Get selected dispersion for a certain station
    resp = DBConn().safequery("select filename,hdfs_path,date,c137,i131 from class where  date=TIMESTAMP \'" +
                datetime.datetime.strftime(results[0], '%m-%d-%Y %H:%M:%S') + "\' and station='" + cln + "';")
    row = resp.fetchone()
    # Check if dispersion has already been turned into GeoJSON (already been cached)
    if (row[3] == None) or (row[4] == None):
        # Save dispersion NetCDF file locally
        urllib.urlretrieve(row[1], row[0])
        # Turn 72 hour disperion into single frame
        dispersion_integral(row[0])
        # Convert dispersion frame to tiff (new tiff still withholds geographical info)
        os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                  row[0] + '\\":C137 ' + row[0].split('.')[0] + '_c137.tiff')
        os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                  row[0] + '\\":I131 ' + row[0].split('.')[0] + '_i131.tiff')
        # Turn HYSPLIT grid into EPSG:4326 projection and use gdal_polygonize
        # in order to turn tiff into GeoJSON
        os.system('make png TIFF_IN=' +
                  row[0].split('.')[0] + '_c137.tiff')
        os.system('make png TIFF_IN=' +
                  row[0].split('.')[0] + '_i131.tiff')
        # Delete used files
        os.system('make clean')
        # Load JSON Files
        with open(APPS_ROOT+ '/' + row[0].split('.')[0] + '_c137.json', 'r') as c137:
            c137_json = json.load(c137)
        with open(APPS_ROOT + '/' + row[0].split('.')[0] + '_i131.json', 'r') as i131:
            i131_json = json.load(i131)
        # Update record for caching purposes
        DBConn().safequery("UPDATE class SET  c137=\'" +
                    json.dumps(c137_json) + "\' WHERE filename=\'" + row[0] + "\'")
        DBConn().safequery("UPDATE class SET  i131=\'" +
                    json.dumps(i131_json) + "\' WHERE filename=\'" + row[0] + "\'")
        # Delete used files
        os.system('rm ' + APPS_ROOT + '/' +
                  row[0].split('.')[0] + '_c137.json')
        os.system('rm ' + APPS_ROOT + '/' +
                  row[0].split('.')[0] + '_i131.json')
        os.system('rm ' + APPS_ROOT + '/' + row[0])
        os.system('rm ' + APPS_ROOT + '/' + 'int_' + row[0])
        # os.system('rm ' + APPS_ROOT + '/' + res[0])
        # Choose selected pollutant
        if pollutant == 'C137':
            dispersion = json.dumps(c137_json)
        else:
            dispersion = json.dumps(i131_json)
        dispersions.append(dispersion)
        scores.append(round(results[1], 3))
    # If dispersion has been cached
    else:
        # os.system('rm ' + APPS_ROOT + '/' + res[0])
        # Choose selected pollutant
        if pollutant == 'C137':
            dispersion = json.dumps(row[3])
        else:
            dispersion = json.dumps(row[4])
        dispersions.append(dispersion)
        scores.append(round(results[1], 3))
    return scores, dispersions

# This function retrieves grid geographical information such as latitude and
# longitude values. Lat/long values are needed in order to create the detection
# maps that are used in order to estimate the location.
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

# This function is the one that is called by the controller when classification models
# have been selected. It uses the above functions in order to function properly.
def cdetections(cur, models, lat_lon, date, pollutant, metric, origin):
    # Load weather variables
    items = load_class_weather(cur, date, origin)
    # Load GRID
    (filelat, filelon, llat, llon) = load_lat_lon(lat_lon)
    # Initialize detection map
    det_obj = Detection(np.zeros(shape=(501, 501)),
                        filelat, filelon, llat, llon)
    det_obj.get_indices()
    det_obj.create_detection_map(resize=True)
    det_map = det_obj._det_map
    # Get prediction
    cl = get_cprediction(det_map, items, models, origin)
    # Get station names
    resp = DBConn().safequery("SELECT station from class group by station order by station;")
    res = resp.fetchall()
    res = [i for i in res]
    print res
    print cl
    class_name = [str(res[i][0]) for i in cl]
    print class_name
    # For each station
    for cln in class_name:
        # Find closest REAL dispersion representation
        (disp_results, weather_results) = calc_scores(cur, items, cln, pollutant, det_map, origin)
        for w in weather_results:
            if w[0] == disp_results[0][0]:
                d = disp_results[0]
                results = (d[0],w[1]*d[1])
        try:
            # Create visualization friendly form of real dispersion
            scores, dispersions = get_disp_frame(cur, cln, pollutant, results)
        except:
            d = disp_results[0]
            results = (d[0],d[1])
            scores, dispersions = get_disp_frame(cur, cln, pollutant, results)
    # Build results as JSON
    scores, dispersions, class_name = zip(
        *sorted(zip(scores, dispersions, class_name), key=lambda k: k[0], reverse=True))
    print scores
    send = {}
    send['stations'] = class_name
    send['scores'] = scores
    send['dispersions'] = dispersions
    return json.dumps(send)

# This function loads weather data, there are two different functions for loading
# weather data due to the fact that clustering and classification methods expect
# weather data in different shape or need different preprocess.
def fbf_load_weather_data(cur, date, origin):
    # Safe query is used due to multiple workers accessing the same database
    # and connection.
    print date
    resp = DBConn().safequery("select filename,hdfs_path,GHT,EXTRACT(EPOCH FROM TIMESTAMP '" +
                date + "' - date)/3600/24 as diff from weather group by date\
                having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    res = resp.fetchone()
    # If model has mult in its title then we need multiple levels (500,700,900 hPa)
    # of the GHT variable.
    if 'mult' in origin:
        # Load weather as a pickled object
        print res[0]
        items = cPickle.loads(str(res[2]))
        items = items.reshape(12,3,64,64)
        # it_shape = items.shape
        # items = np.average(items,0)
        # items = items.reshape(1, 1, 1, it_shape[
        #                       1], it_shape[2], it_shape[3])
    else:
        # GHT 700hPa
        items = cPickle.loads(str(res[2]))
        items = items[:, 1, :, :]
        it_shape = items.shape
        items = np.average(items,0)
        items = items.reshape(
            1, 1, it_shape[0], 1, it_shape[1], it_shape[2])
    return items,res

def load_weather_data(cur, date, origin):
    # Safe query is used due to multiple workers accessing the same database
    # and connection.
    resp = DBConn().safequery("select filename,hdfs_path,GHT,EXTRACT(EPOCH FROM TIMESTAMP '" +
                date + "' - date)/3600/24 as diff from weather group by date\
                having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    res = resp.fetchone()
    # If model has mult in its title then we need multiple levels (500,700,900 hPa)
    # of the GHT variable.
    if 'mult' in origin:
        # Load weather as a pickled object
        items = cPickle.loads(str(res[2]))
        it_shape = items.shape
        items = np.average(items,0)
        items = items.reshape(1, 1, 1, it_shape[
                              1], it_shape[2], it_shape[3])
        # items = items.reshape(1, 1, items.shape[0], items.shape[
        #                       1], items.shape[2], items.shape[3])
    else:
        # GHT 700hPa
        items = cPickle.loads(str(res[2]))
        items = items[:, 1, :, :]
        items = items.reshape(
            1, 1, items.shape[0], 1, items.shape[1], items.shape[2])
    return items,res

def fbf(current_weather):
    global_dist=[]
    for d in os.listdir(APPS_ROOT):
        if d.endswith('.npy'):
           descriptor = np.load(d).reshape(14,3,64,64)
           local_dists = []
           for i in xrange(len(current_weather)):
               for lvl in xrange(3):
                       local_dists.append(np.linalg.norm(current_weather[i,lvl,:]-descriptor[i,lvl,:]))
           local_dists = np.array(local_dists)
           local_dists = np.mean(local_dists)
           global_dist.append((d.split('.npy')[0],local_dists))
    global_dist = sorted(global_dist, key=lambda x: x[1], reverse=False)
    return global_dist

# This function return the closest cluster based on the real weather choosen by
# the user.
def load_cluster_date(items, models, origin):
    # Convert weather data to 2 dimensions
    ds = Dataset_transformations(items, 1000, items.shape)
    x = items.shape[2]
    y = items.shape[3]
    ds.twod_transformation()
    # Normalization
    ds.normalize()
    # Find selected model
    for m in models:
        # If anything else then k-means, then we need to re run the clustering
        # model and get the output of the hidden layer for centroid comparison
        if origin == m[0]:
            if 'kmeans' not in m[0]:
                clust_obj = m[2]
                ds._items = m[1].get_hidden(ds._items.T)
                # Select closest centroid
                cd = clust_obj.centroids_distance(ds, features_first=False)
                # Return centroid as date
                cluster_date = utils.reconstruct_date(
                    clust_obj._desc_date[cd[0][0]])
            # If kmeans was selected, then perform centroid comparsion in raw data
            else:
                clust_obj = m[1]
                cd = clust_obj.centroids_distance(ds, features_first=True)
                cluster_date = utils.reconstruct_date(
                    clust_obj._desc_date[cd[0][0]])
    cd = [(utils.reconstruct_date(clust_obj._desc_date[c[0]]),c[1]) for c in cd]
    cd = sorted(cd, key=lambda k: k[1],reverse=False)
    print cd
    return cluster_date

def fbf_load_cluster_date(items, models, origin):
    for m in models:
        if origin == m[0]:
            if 'kmeans' not in m[0]:
                clust_obj = m[2]
            else:
                clust_obj = m[1]
    cd = []
    cdd = fbf(items)
    cd_scale = np.array(cdd)
    cd_scale = cd_scale[:,1].astype(np.float32)
    cd_scale = cd_scale / np.max(cd_scale)
    print cd_scale
    for pos,i in enumerate(cdd):
        cd.append((cdd[pos][0],cd_scale[pos]))
    # cd = [(utils.reconstruct_date(clust_obj._desc_date[c[0]]),c[1]) for c in cd]
    # cd = sorted(cd, key=lambda k: k[1],reverse=False)
    return cd[0][0],cd

def calc_station_scores(cur, lat_lon, timestamp, origin, descriptor, pollutant):
    # Load GRID
    (filelat, filelon, llat, llon) = load_lat_lon(lat_lon)
    results = []
    # Get all dispersion of every station for a certain cluster
    res = DBConn().safequery("select filename,hdfs_path,station,c137_pickle,i131_pickle from cluster where date=TIMESTAMP \'" +
                datetime.datetime.strftime(timestamp, '%m-%d-%Y %H:%M:%S') + "\' and origin='" + origin + "' and descriptor='" + descriptor + "'")
    # For each dispersion/station
    for row in res:
        # Create detection point map
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
        # Compare detection map to real dispersion
        # if det_obj.calc() != 0:
        results.append((row[2], det_obj.cosine()))
        # else:
        #    results.append((row[2], 0))
    # Return all results
    return results

# This functions calculates scores for each station of a certain cluster by comparing
# every dispersion with the constructed detection point map.
def fbf_calc_station_scores(cur, lat_lon, timestamp, origin, descriptor, pollutant, cluster_dates):
    # Load GRID
    (filelat, filelon, llat, llon) = load_lat_lon(lat_lon)
    det_obj = Detection(np.zeros(shape=(501, 501)),
                        filelat, filelon, llat, llon)
    det_obj.get_indices()
    det_obj.create_detection_map(resize=False)
    det_map = det_obj._det_map
    results = []
    new_cd = []
    for cd in cluster_dates:
        print cd[0],cd[1]
        cluster_date = datetime.datetime.strptime(cd[0],'%y-%m-%d-%H')
        cluster_date = datetime.datetime.strftime(cluster_date, '%m-%d-%Y %H:%M:%S')
        resp = DBConn().safequery(
            "SELECT station,hdfs_path,c137_pickle,i131_pickle from cluster where date=TIMESTAMP \'" + cluster_date + "\';")
        res = resp.fetchall()
        disp_results = []
        for row in res:
            if pollutant == 'C137':
                det = cPickle.loads(str(row[2]))
            else:
                det = cPickle.loads(str(row[3]))
            # Preprocessing of dispersion
            det_shape = det.shape
            det = maxabs_scale(det.flatten(),axis=1)
            det = det.reshape(det_shape)
            # Get distance between real dispersion and detection points
            disp_results.append((row[0], scipy.spatial.distance.cosine(det.flatten(), det_map.flatten())))
        results = sorted(disp_results, key=lambda k: k[1], reverse=False)
        for r in results:
            if (r[1] < 1.0) and (r[0] != 'GARONA' and r[0] != 'IGNALINA'):
                new_cd.append((cd[0]+'_'+r[0],r[1]))
        # print results[0]
        # new_cd.append((cd[0],cd[1]*results[0]))
    new_cd = sorted(new_cd, key=lambda k: k[1] , reverse=False)
    print new_cd
    new_cd2 = []
    for c in new_cd:
        if not new_cd2:
            new_cd2.append(c)
        else:
            test = [i[0].split('_')[1] for i in new_cd2]
            if not(c[0].split('_')[1] in test):
                new_cd2.append(c)
    print '------------------------------------------------'
    return new_cd2
    # Get all dispersion of every station for a certain cluster
    # res = DBConn().safequery("select filename,hdfs_path,station,c137_pickle,i131_pickle from cluster where date=TIMESTAMP \'" +
    #             datetime.datetime.strftime(timestamp, '%m-%d-%Y %H:%M:%S') + "\' and origin='" + origin + "' and descriptor='" + descriptor + "'")
    # # For each dispersion/station
    # for row in res:
    #     # Create detection point map
    #     if pollutant == 'C137':
    #         det_obj = Detection(cPickle.loads(
    #             str(row[3])), filelat, filelon, llat, llon)
    #         det_obj.get_indices()
    #         det_obj.create_detection_map()
    #     else:
    #         det_obj = Detection(cPickle.loads(
    #             str(row[4])), filelat, filelon, llat, llon)
    #         det_obj.get_indices()
    #         det_obj.create_detection_map()
    #     # Compare detection map to real dispersion
    #     if det_obj.calc() != 0:
    #         results.append((row[2], det_obj.cosine()))
    #     else:
    #         results.append((row[2], 0))
    # # Return all results
    # return results


# This function selects the top 3 most close stations to a dispersion and converts
# their dispersions into GeoJSON form for visualization
def get_top3_stations(cur, top3, timestamp, origin, pollutant):
    # Get top 3 names and scores
    top3_names = [top[0] for top in top3]
    top3_scores = [round(top[1], 3) for top in top3]
    stations = []
    scores = []
    dispersions = []
    # Find their database records
    resp = DBConn().safequery("select filename,hdfs_path,station,c137,i131 from cluster where date=TIMESTAMP \'" +
                datetime.datetime.strftime(timestamp, '%m-%d-%Y %H:%M:%S') + "\' and origin='" + origin + "'")
    rows = resp.fetchall()
    # For each dispersion in a certain cluster
    for row in rows:
        # If the name is in top 3
        if row[2] in top3_names:
            # Check if dispersion visualization form has already been cached
            if (row[3] == None) or (row[4] == None):
                # If not, retrieve original NetCDF file
                urllib.urlretrieve(row[1], row[0])
                # Convert 72 hour dispersion into a single frame
                dispersion_integral(row[0])
                # Convert dispersion frame to tiff (new tiff still withholds geographical info)
                os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                          row[0] + '\\":C137 ' + row[0].split('.')[0] + '_c137.tiff')
                os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                          row[0] + '\\":I131 ' + row[0].split('.')[0] + '_i131.tiff')
                # Turn HYSPLIT grid into EPSG:4326 projection and use gdal_polygonize
                # in order to turn tiff into GeoJSON
                os.system('make png TIFF_IN=' +
                          row[0].split('.')[0] + '_c137.tiff')
                os.system('make png TIFF_IN=' +
                          row[0].split('.')[0] + '_i131.tiff')
                # Delete used files
                os.system('make clean')
                # Load JSON Files
                with open(row[0].split('.')[0] + '_c137.json', 'r') as c137:
                    c137_json = json.load(c137)
                with open(row[0].split('.')[0] + '_i131.json', 'r') as i131:
                    i131_json = json.load(i131)
                # Update record for caching purposes
                DBConn().safequery("UPDATE cluster SET  c137=\'" +
                            json.dumps(c137_json) + "\' WHERE filename=\'" + row[0] + "\'")
                DBConn().safequery("UPDATE cluster SET  i131=\'" +
                            json.dumps(i131_json) + "\' WHERE filename=\'" + row[0] + "\'")
                # Delete used files
                os.system('rm ' + APPS_ROOT + '/' +
                          row[0].split('.')[0] + '_c137.json')
                os.system('rm ' + APPS_ROOT + '/' +
                          row[0].split('.')[0] + '_i131.json')
                os.system('rm ' + APPS_ROOT + '/' + row[0])
                os.system('rm ' + APPS_ROOT + '/' + 'int_' + row[0])
                # os.system('rm ' + APPS_ROOT + '/' + res[0])
                stations.append(str(row[2]))
                scores.append(top3_scores[top3_names.index(row[2])])
                # Choose selected pollutant
                if pollutant == 'C137':
                    dispersions.append(json.dumps(c137_json))
                else:
                    dispersions.append(json.dumps(i131_json))
            # If dispersion has been cached
            else:
                # os.system('rm ' + APPS_ROOT + '/' + res[0])
                stations.append(str(row[2]))
                scores.append(top3_scores[top3_names.index(row[2])])
                if pollutant == 'C137':
                    dispersions.append(json.dumps(row[3]))
                else:
                    dispersions.append(json.dumps(row[4]))
    return scores, dispersions, stations

# This function selects the top 3 most close stations to a dispersion and converts
# their dispersions into GeoJSON form for visualization
def fbf_get_top3_stations(cur, top3, timestamp, origin, pollutant):
    # Get top 3 names and scores
    top3_names = [top[0] for top in top3]
    top3_scores = [round(top[1], 3) for top in top3]
    stations = []
    scores = []
    dispersions = []
    # Find their database records
    for t in top3:
        localtime = datetime.datetime.strptime(t[0].split('_')[0], '%y-%m-%d-%H')
        resp = DBConn().safequery("select filename,hdfs_path,station,c137,i131 from cluster where date=TIMESTAMP \'" +
                    datetime.datetime.strftime(localtime, '%m-%d-%Y %H:%M:%S') + "\' and origin='" + origin + "' and station='"+t[0].split('_')[1]+"'")
        rows = resp.fetchall()
        # For each dispersion in a certain cluster
        for row in rows:
            # If the name is in top 3
            # if row[2] in top3_names:
                # Check if dispersion visualization form has already been cached
            if (row[3] == None) or (row[4] == None):
                # If not, retrieve original NetCDF file
                urllib.urlretrieve(row[1], row[0])
                # Convert 72 hour dispersion into a single frame
                dispersion_integral(row[0])
                # Convert dispersion frame to tiff (new tiff still withholds geographical info)
                os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                          row[0] + '\\":C137 ' + row[0].split('.')[0] + '_c137.tiff')
                os.system('gdal_translate NETCDF:\\"' + APPS_ROOT + '/' + 'int_' +
                          row[0] + '\\":I131 ' + row[0].split('.')[0] + '_i131.tiff')
                # Turn HYSPLIT grid into EPSG:4326 projection and use gdal_polygonize
                # in order to turn tiff into GeoJSON
                os.system('make png TIFF_IN=' +
                          row[0].split('.')[0] + '_c137.tiff')
                os.system('make png TIFF_IN=' +
                          row[0].split('.')[0] + '_i131.tiff')
                # Delete used files
                os.system('make clean')
                # Load JSON Files
                with open(row[0].split('.')[0] + '_c137.json', 'r') as c137:
                    c137_json = json.load(c137)
                with open(row[0].split('.')[0] + '_i131.json', 'r') as i131:
                    i131_json = json.load(i131)
                # Update record for caching purposes
                DBConn().safequery("UPDATE cluster SET  c137=\'" +
                            json.dumps(c137_json) + "\' WHERE filename=\'" + row[0] + "\'")
                DBConn().safequery("UPDATE cluster SET  i131=\'" +
                            json.dumps(i131_json) + "\' WHERE filename=\'" + row[0] + "\'")
                # Delete used files
                os.system('rm ' + APPS_ROOT + '/' +
                          row[0].split('.')[0] + '_c137.json')
                os.system('rm ' + APPS_ROOT + '/' +
                          row[0].split('.')[0] + '_i131.json')
                os.system('rm ' + APPS_ROOT + '/' + row[0])
                os.system('rm ' + APPS_ROOT + '/' + 'int_' + row[0])
                # os.system('rm ' + APPS_ROOT + '/' + res[0])
                stations.append(str(row[2]))
                # scores.append(top3_scores[top3_names.index(row[2])])
                scores.append(round(t[1],3))
                # Choose selected pollutant
                if pollutant == 'C137':
                    dispersions.append(json.dumps(c137_json))
                else:
                    dispersions.append(json.dumps(i131_json))
            # If dispersion has been cached
            else:
                # os.system('rm ' + APPS_ROOT + '/' + res[0])
                stations.append(str(row[2]))
                # scores.append(top3_scores[top3_names.index(row[2])])
                scores.append(round(t[1],3))
                if pollutant == 'C137':
                    dispersions.append(json.dumps(row[3]))
                else:
                    dispersions.append(json.dumps(row[4]))
    return scores, dispersions, stations

# This function is the one that is called by the controller when clustering models
# have been selected. It uses the above functions in order to function properly.
def fbf_detections(cur, models, lat_lon, date, pollutant, metric, origin):
    # Load weather variable
    (items,res) = fbf_load_weather_data(cur, date, origin)
    # Get best cluster candidate
    (cluster_date,cluster_dates) = fbf_load_cluster_date(items, models, origin)
    descriptor = origin.split('_')
    descriptor = descriptor[len(descriptor) - 1]
    timestamp = datetime.datetime.strptime(cluster_date, '%y-%m-%d-%H')
    # Get scores for each station for the best cluster candidate
    results = fbf_calc_station_scores(cur, lat_lon, timestamp, origin, descriptor, pollutant, cluster_dates)
    # Sort scores
    # results = sorted(results, key=lambda k: k[1] if k[
    #                  1] > 0 else float('inf'), reverse=False)
    # Get top 3 stations
    # top3 = results[:3]
    top3 = results
    # print top3
    # Turn top 3 station dispersion to visualization friendly form
    scores, dispersions, stations = fbf_get_top3_stations(cur, top3, timestamp, origin, pollutant)
    # Convert results to JSON form
    scores, dispersions, stations = zip(
        *sorted(zip(scores, dispersions, stations), key=lambda k: k[0] if k[0] > 0 else float('inf'), reverse=False))
    send = {}
    send['stations'] = stations
    send['scores'] = scores
    send['dispersions'] = dispersions
    return json.dumps(send)

# This function is the one that is called by the controller when clustering models
# have been selected. It uses the above functions in order to function properly.
def detections(cur, models, lat_lon, date, pollutant, metric, origin):
    # Load weather variable
    (items,res) = load_weather_data(cur, date, origin)
    # Get best cluster candidate
    cluster_date = load_cluster_date(items, models, origin)
    descriptor = origin.split('_')
    descriptor = descriptor[len(descriptor) - 1]
    timestamp = datetime.datetime.strptime(cluster_date, '%y-%m-%d-%H')
    # Get scores for each station for the best cluster candidate
    results = calc_station_scores(cur, lat_lon, timestamp, origin, descriptor, pollutant)
    # Sort scores
    results = sorted(results, key=lambda k: k[1] if k[
                     1] > 0 else float('inf'), reverse=False)
    print results
    top3 = results
    # Get top 3 stations
    # top3 = results[:3]
    # print top3
    # Turn top 3 station dispersion to visualization friendly form
    scores, dispersions, stations = get_top3_stations(cur, top3, timestamp, origin, pollutant)
    # Convert results to JSON form
    scores, dispersions, stations = zip(
        *sorted(zip(scores, dispersions, stations), key=lambda k: k[0] if k[0] > 0 else float('inf'), reverse=False))
    send = {}
    send['stations'] = stations
    send['scores'] = scores
    send['dispersions'] = dispersions
    return json.dumps(send)

# This function returns all ingested models/methods available to the user.
def get_methods(cur):
    res = DBConn().safequery("select origin,html from models;")
    origins = []
    for row in res:
        origin = {}
        origin['html'] = row[1]
        origin['origin'] = row[0]
        origins.append(origin)
    return json.dumps(origins)

# This method return the best real weather data candidate for a given date
def get_closest_weather(cur, date, level):
    level = int(level)
    # Query based on weather pressure level and date
    if level == 22:
        resp = DBConn().safequery("select filename,hdfs_path,wind_dir500,EXTRACT(EPOCH FROM TIMESTAMP '" +
                    date + "' - date)/3600/24 as diff from weather group by date\
                    having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    elif level == 26:
        resp = DBConn().safequery("select filename,hdfs_path,wind_dir700,EXTRACT(EPOCH FROM TIMESTAMP '" +
                    date + "' - date)/3600/24 as diff from weather group by date\
                    having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    elif level == 33:
        resp = DBConn().safequery("select filename,hdfs_path,wind_dir900,EXTRACT(EPOCH FROM TIMESTAMP '" +
                    date + "' - date)/3600/24 as diff from weather group by date\
                    having EXTRACT(EPOCH FROM TIMESTAMP '" + date + "' - date)/3600/24 >= 0 order by diff;")
    res = resp.fetchone()
    # Check if date result is in bounds
    if res[3] > 5:
        return json.dumps({'error': 'date is out of bounds'})
    # Check if weather data has already been cached
    if res[2] == None:
        # If not, get original NetCDF file
        urllib.urlretrieve(res[1], res[0])
        # Convert NetCDF to GeoJSON format for visualization purposes
        json_dir = calc_winddir(res[0], level)
        # Delete used files
        os.system('rm ' + APPS_ROOT + '/' + res[0])
        # Update database record, therefore cache the visualization format for
        # this weather file
        if level == 22:
            DBConn().safequery("UPDATE weather SET  wind_dir500=\'" +
                        json_dir + "\' WHERE filename=\'" + res[0] + "\'")
        elif level == 26:
            DBConn().safequery("UPDATE weather SET  wind_dir700=\'" +
                        json_dir + "\' WHERE filename=\'" + res[0] + "\'")
        elif level == 33:
            DBConn().safequery("UPDATE weather SET  wind_dir900=\'" +
                        json_dir + "\' WHERE filename=\'" + res[0] + "\'")
        return json_dir
    # If already cached
    else:
        return json.dumps(res[2])

# This function loads the dispersion grid used in combination with SEMAGROW in
# order to return affected areas. The difference with the load_lat_lon is that
# this grid is split into cells rather than (lat,lon) points.
def load_gridcells():
    with open('dispersion_grid.json') as ff:
         cells = json.load(ff)
    cell_pols = []
    for cell in cells:
        points = []
        points.append(Point(float(cell['bottom_left']['lon']),float(cell['bottom_left']['lat'])))
        points.append(Point(float(cell['top_left']['lon']),float(cell['top_left']['lat'])))
        points.append(Point(float(cell['top_right']['lon']),float(cell['top_right']['lat'])))
        points.append(Point(float(cell['bottom_right']['lon']),float(cell['bottom_right']['lat'])))
        # Convert them into Shapely Polygons for convinient manipulation
        pol = Polygon([[p.x, p.y] for p in points])
        cell_pol = {}
        cell_pol['id'] = cell['id']
        cell_pol['obj'] = pol
        cell_pols.append(cell_pol)
    return cell_pols

# This function is used in order to log the computation time of various functions
def timing(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


from SPARQLWrapper import SPARQLWrapper, JSON

# This function performs the necessary queries in order to get affected areas,
# given an endpoint and a single id of the grid cell
def single_query(semagrow, cell_id):
  semagrow.setQuery("""
  PREFIX  strdf: <http://strdf.di.uoa.gr/ontology#>

  SELECT  ?geoname ?lat ?long ?name ?population
  WHERE
    { <http://iit.demokritos.gr/%s> strdf:hasGeometry ?geometry .
      ?geoname  <http://www.opengis.net/ont/geosparql#asWKT>  ?point ;
                <http://www.w3.org/2003/01/geo/wgs84_pos#lat>  ?lat ;
                <http://www.w3.org/2003/01/geo/wgs84_pos#long>  ?long ;
                <http://www.geonames.org/ontology#name> ?name ;
                <http://www.geonames.org/ontology#population>  ?population .
      FILTER strdf:within(?point, ?geometry)
    }
  """%cell_id)

  semagrow.setReturnFormat(JSON)

  results = semagrow.queryAndConvert()

  return results

# This function performs the necessary queries in order to get affected areas,
# given an endpoint and multiple cell id's (batches).
def query(semagrow, cell_id):
  values = ""
  for id in cell_id:
    values = values + "<http://iit.demokritos.gr/"+str(id)+"> "
  semagrow.setQuery("""
  PREFIX  strdf: <http://strdf.di.uoa.gr/ontology#>

  SELECT  ?geoname ?lat ?long ?name ?population
  WHERE
    { ?cellid strdf:hasGeometry ?geometry .
      ?geoname  <http://www.opengis.net/ont/geosparql#asWKT>  ?point ;
                <http://www.w3.org/2003/01/geo/wgs84_pos#lat>  ?lat ;
                <http://www.w3.org/2003/01/geo/wgs84_pos#long>  ?long ;
                <http://www.geonames.org/ontology#name> ?name ;
                <http://www.geonames.org/ontology#population>  ?population .
      VALUES ?cellid { %s }
      FILTER strdf:within(?point, ?geometry)
    }
  """%values)

  semagrow.setReturnFormat(JSON)

  results = semagrow.queryAndConvert()

  return results

# This is the function called by the controller in order to return affected from
# the dispersion areas (when querying each id individually).
def single_pop(cell_pols,disp):
    start = time.time()
    # Load dispersion in the JSON format
    disp = json.loads(disp)
    # Turn JSON into shapely polygons
    multi = MultiPolygon([shape(pol['geometry']) for pol in disp['features']])
    # Get intersection with our grid, therefore which cells are being affected_ids
    # by the dispersion
    affected_ids = [pol['id'] for pol in cell_pols if multi.intersects(pol['obj'])]
    # Remove duplicate entries
    affected_ids = list(set(affected_ids))
    multi_points = []
    # For each id query SEMAGROW to get more info
    for id in affected_ids:
        try:
            semagrow = SPARQLWrapper('http://10.0.10.12:9999/SemaGrow/query')
            results = single_query(semagrow,id)
            points = [(Point(float(res['long']['value']),float(res['lat']['value'])),int(res['population']['value']),res['geoname']['value'],res['name']['value']) for res in results['results']['bindings']]
            multi_points.append(points)
        except:
            pass
    # Collapse multi points into single list
    multi_points = list(chain.from_iterable(multi_points))
    jpols = []
    timing(start,time.time())
    start = time.time()
    # Build response in JSON format
    for p,point in enumerate(multi_points):
        jpols.append(dict(type='Feature', properties={"POP":unicode(point[1]),"URI":unicode(point[2]),"NAME":unicode(point[3])}, geometry=mapping(point[0])))
    end_res = dict(type='FeatureCollection', crs={ "type": "name", "properties": { "name":"urn:ogc:def:crs:OGC:1.3:CRS84" }},features=jpols)
    timing(start,time.time())
    return json.dumps(end_res)

# This is the function called by the controller in order to return affected from
# the dispersion areas (batch id querying).
def pop(cell_pols,disp):
    start = time.time()
    # Load dispersion in the JSON format
    disp = json.loads(disp)
    # Turn JSON into shapely polygons
    multi = MultiPolygon([shape(pol['geometry']) for pol in disp['features']])
    # Get intersection with our grid, therefore which cells are being affected_ids
    # by the dispersion
    affected_ids = [pol['id'] for pol in cell_pols if multi.intersects(pol['obj'])]
    # Remove duplicate entries
    affected_ids = list(set(affected_ids))
    multi_points = []
    # Open endpoint
    semagrow = SPARQLWrapper('http://10.0.10.12:9999/SemaGrow/query')
    # Batch query
    for batch in range(0,len(affected_ids),semagrow_batch_size):
        results = query(semagrow,affected_ids[batch:batch+semagrow_batch_size])
        points = [(Point(float(res['long']['value']),float(res['lat']['value'])),int(res['population']['value']),res['geoname']['value'],res['name']['value']) for res in results['results']['bindings']]
        multi_points.append(points)
    # Collapse multi points into single list
    multi_points = list(chain.from_iterable(multi_points))
    jpols = []
    timing(start,time.time())
    # Build response in JSON format
    for p,point in enumerate(multi_points):
        jpols.append(dict(type='Feature', properties={"POP":unicode(point[1]),"URI":unicode(point[2]),"NAME":unicode(point[3])}, geometry=mapping(point[0])))
    end_res = dict(type='FeatureCollection', crs={ "type": "name", "properties": { "name":"urn:ogc:def:crs:OGC:1.3:CRS84" }},features=jpols)
    return json.dumps(end_res)
