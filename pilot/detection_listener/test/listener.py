from web import app
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import api_methods
import getpass
import psycopg2
import os
import dataset_utils as utils
import urllib
from celery import Celery

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# BOOTSTRAP_SERVE_LOCAL = True
# app = Flask(__name__)
# CORS(app)
#
# app.config.from_object(__name__)


inp = None
parameters = None
export_template = None
clust_obj = None
exper = None
conn = None
cur = None
dpass = getpass.getpass()
APPS_ROOT = os.path.dirname(os.path.abspath(__file__))

@celery.task(bind=True)
def go_async(self, lat_lon, date, pollutant, metric, origin):
    return api_methods.cdetections(cur, models, lat_lon, date, pollutant, metric, origin)


@app.route('/class_detections/<date>/<pollutant>/<metric>/<origin>', methods=['POST'])
def cdetections(date, pollutant, metric, origin):
    lat_lon = request.get_json(force=True)
    task = go_async.apply_async(lat_lon, date, pollutant, metric, origin)
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        // job did not start yet
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

if __name__ == '__main__':
    with open('db_info.json', 'r') as data_file:
        dbpar = json.load(data_file)
    conn = psycopg2.connect("dbname='" + dbpar['dbname'] + "' user='" + dbpar['user'] +
                            "' host='" + dbpar['host'] + "' port='" + dbpar['port'] + "'password='" + dpass + "'")
    cur = conn.cursor()
    inp = 'parameters.json'
    models = []
    cur.execute("SELECT * from models")
    for row in cur:
        print row[1]
        urllib.urlretrieve(row[2], row[1])
        config = utils.load(row[1])
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
        os.system('rm ' + APPS_ROOT + '/' + row[1])
    try:
        app.run(host='0.0.0.0')
    except Exception:
        pass
