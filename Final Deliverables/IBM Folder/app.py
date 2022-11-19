import flask
from flask import request, render_template
from flask_cors import CORS
import joblib

app = flask.Flask(__name__, static_url_path='')
CORS(app)

@app.route('/', methods=['GET'])
def sendHomePage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predictSpecies():
    sl = float(request.form['sl'])
    sw = float(request.form['sw'])
    pl = float(request.form['pl'])
    pw = float(request.form['pw'])
    X = [[sl, sw, pl, pw]]
    model = joblib.load('model.pkl')
    species = model.predict(X)[0]
    return render_template('predict.html',predict=species)

if __name__ == '__main__':
    app.run()