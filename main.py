from flask import Flask
from bp_main import bp_main


app = Flask(__name__)


app.register_blueprint(bp_main, url_prefix='/')

if __name__ == '__main__':
    app.run(debug=True)