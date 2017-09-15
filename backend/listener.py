"""
   SCRIPT INFO
   ---------------------------------------------------------------------------
   This script acts a controller for the SC5 #3 pilot. It controls the
   flow of the UI by calling the appropiate functions and returning their
   respective results.
   ---------------------------------------------------------------------------
"""

from web import app
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS, cross_origin
import json
import api_methods
import getpass
import psycopg2
import os
import dataset_utils as utils
import urllib
from celery import Celery
import base64

# Initialize and configure FLASK parameters
BOOTSTRAP_SERVE_LOCAL = True
app = Flask(__name__)
CORS(app)

# Configre Celery parameters
app.config.from_object(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Initialize global variables
parameters = None
export_template = None
clust_obj = None
exper = None
conn = None
cur = None
cell_pols = None
APPS_ROOT = os.path.dirname(os.path.abspath(__file__))

# Celery async task for classifications methods, classifications needs to be an
# asynchronous task due to response time that freezing the UI
@celery.task(bind=True)
def go_async(self, lat_lon, date, pollutant, metric, origin):
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': api_methods.cdetections(cur, models, lat_lon, date, pollutant, metric, origin)}

# Service handling the classification methods
@app.route('/class_detections/<date>/<pollutant>/<metric>/<origin>', methods=['POST'])
def cdetections(date, pollutant, metric, origin):
    lat_lon = request.get_json(force=True)
    task = go_async.apply_async(args=[lat_lon, date, pollutant, metric, origin])
    response = {
        'id': task.id
    }
    return jsonify(response)

# Service that checks the state of an ansychronous task running in the background.
# This service is used for monitoring the tasks use for source estimation using
# classification methods and detecting the affected population using SEMAGROW
@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = go_async.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),
        }
    return jsonify(response)

# Service handling the clustering methods
@app.route('/detections/<date>/<pollutant>/<metric>/<origin>', methods=['POST'])
def detections(date, pollutant, metric, origin):
    lat_lon = request.get_json(force=True)
    return api_methods.detections(cur, models, lat_lon, date, pollutant, metric, origin)

# Service that returns the ingested models/methods for source estimation
@app.route('/getMethods/', methods=['GET'])
def getMethods():
    return api_methods.get_methods(cur)

# Service that returns the closest weather represenatation based on the input
# given as a date. Weather represenatation is used for visualization
@app.route('/getClosestWeather/<date>/<level>', methods=['GET'])
def getClosestWeather(date, level):
    return api_methods.get_closest_weather(cur, date, level)

# Celery async task for affected population detection
@app.route('/population/', methods=['POST'])
def population():
    disp = request.get_json(force=True)
    task = class_async.apply_async(args=[disp])
    response = {
        'id': task.id
    }
    return json.dumps(response)

# Service handling the  affected population detection
@celery.task(bind=True)
def class_async(self, disp):
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': api_methods.pop(cell_pols,disp)}

# Pseudo main, these lines are global due to the fact that this script returns
# using Gunicorn, when relocating these lines to a function the models do not
# load propely
from dbconn import DBConn
cur = DBConn().engine
models = []
res = cur.execute("SELECT * from models")
for row in res:
    urllib.urlretrieve(row[2], str(os.getpid())+row[1])
    print row[1]
    config = utils.load(str(os.getpid())+row[1])
    m = config.next()
    try:
        c = config.next()
    except:
        c = m
    current = [mod[1] for mod in models]
    try:
        pos = current.index(m)
        models.append((row[0], models[pos][1], c))
    except:
        models.append((row[0], m, c))
    os.system('rm ' + APPS_ROOT + '/' + str(os.getpid())+row[1])
print 'Loading grid cells.......'
cell_pols = api_methods.load_gridcells()
print 'Done.......'

# Running and binding port/host. Only usable if listener script runs with the
# default WSGI Server and not with Gunicorn
if __name__ == '__main__':
    app.run(host='0.0.0.0')
