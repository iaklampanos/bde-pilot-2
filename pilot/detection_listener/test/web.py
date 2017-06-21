from flask import Flask

# global options
BOOTSTRAP_SERVE_LOCAL = True
# create our application :)
app = Flask(__name__)

app.config.from_object(__name__)
