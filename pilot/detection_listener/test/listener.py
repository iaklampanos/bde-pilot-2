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

BOOTSTRAP_SERVE_LOCAL = True
app = Flask(__name__)
CORS(app)

app.config.from_object(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

inp = None
parameters = None
export_template = None
clust_obj = None
exper = None
conn = None
cur = None
# dpass = getpass.getpass()
APPS_ROOT = os.path.dirname(os.path.abspath(__file__))

@celery.task(bind=True)
def go_async(self, lat_lon, date, pollutant, metric, origin):
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': api_methods.cdetections(cur, models, lat_lon, date, pollutant, metric, origin)}


@app.route('/class_detections/<date>/<pollutant>/<metric>/<origin>', methods=['POST'])
def cdetections(date, pollutant, metric, origin):
    lat_lon = request.get_json(force=True)
    task = go_async.apply_async(args=[lat_lon, date, pollutant, metric, origin])
    response = {
        'id': task.id
    }
    return jsonify(response)

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
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/detections/<date>/<pollutant>/<metric>/<origin>', methods=['POST'])
def detections(date, pollutant, metric, origin):
    lat_lon = request.get_json(force=True)
    return api_methods.detections(cur, models, lat_lon, date, pollutant, metric, origin)

@app.route('/getMethods/', methods=['GET'])
def getMethods():
    return api_methods.get_methods(cur)


@app.route('/getClosestWeather/<date>/<level>', methods=['GET'])
def getClosestWeather(date, level):
    return api_methods.get_closest(cur, date, level)

with open('db_info.json', 'r') as data_file:
     dbpar = json.load(data_file)
# conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
#                         "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + base64.b64decode(dbpar['pass']) + "'")
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://'+dbpar['user']+':'+base64.b64decode(dbpar['pass'])+'@'+dbpar['host']+'/'+dbpar['dbname']+'')
engine.connect()
cur = engine
from multiprocessing.util import register_after_fork
register_after_fork(engine, engine.dispose)
inp = 'parameters.json'
models = []
res = cur.execute("SELECT * from models")
for row in res:
    urllib.urlretrieve(row[2], str(os.getpid())+row[1])
    print row[2]
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

if __name__ == '__main__':
    app.run(host='0.0.0.0')
