from web import app
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import api_methods

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
dpass = getpass.getpass()
APPS_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/class_detections/<date>/<pollutant>/<metric>/<origin>', methods=['POST'])
api_methods.cdetections(date, pollutant, metric, origin)


@app.route('/detections/<date>/<pollutant>/<metric>/<origin>', methods=['POST'])
api_methods.detections(date, pollutant, metric, origin)

@app.route('/getMethods/', methods=['GET'])
api_methods.get_methods()


@app.route('/getClosestWeather/<date>/<level>', methods=['GET'])
api_methods.get_closest(date, level)

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
