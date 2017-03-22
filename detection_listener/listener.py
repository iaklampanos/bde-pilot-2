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

BOOTSTRAP_SERVE_LOCAL = True
app = Flask(__name__)
CORS(app)

app.config.from_object(__name__)


inp = None
parameters = None
export_template = None
clust_obj = None
exper = None

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
    for station in parameters['stations']:
        nc_file = Dataset(
            path + '/' + station['name'] + '-' + cluster_date + '.nc', 'r')
        filelat = nc_file.variables['latitude'][:]
        filelon = nc_file.variables['longitude'][:]
        det_obj = Detection(nc_file, pollutant, filelat, filelon, llat, llon)
        det_obj.get_indices()
        det_obj.calculate_concetration()
        det_obj.create_detection_map()
        results.append((station['name'], det_obj.calc_score()))
    results = sorted(results, key=lambda k: k[1], reverse=True)
    send = {}
    send["station"] = str(results[0][0])
    send["date"] = str(utils.reconstruct_date(clust_obj._desc_date[cd[0][0]]))
    send["pollutant"] = str(pollutant)
    return json.dumps(send)


@app.route('/test_files', methods=['GET'])
def l():
    return json.dumps(sorted(os.listdir(parameters['test_netcdf_path'])))


if __name__ == '__main__':
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
