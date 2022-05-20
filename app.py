from flask import Flask, render_template, request
from joblib import load
import numpy as np
from os import environ

app = Flask(__name__)

TEAMS = {
    'Chennai Super Kings': {
        'name': 'Chennai Super Kings',
        'value': [1, 0, 0, 0, 0, 0, 0, 0]
    },
    'Delhi Daredevils': {
        'name': 'Delhi Daredevils',
        'value': [0, 1, 0, 0, 0, 0, 0, 0]
    },
    'Kings XI Punjab': {
        'name': 'Kings XI Punjab',
        'value': [0, 0, 1, 0, 0, 0, 0, 0]
    },
    'Kolkata Knight Riders': {
        'name': 'Kolkata Knight Riders',
        'value': [0, 0, 0, 1, 0, 0, 0, 0]
    },
    'Mumbai Indians': {
        'name': 'Mumbai Indians',
        'value': [0, 0, 0, 0, 1, 0, 0, 0]
    },
    'Rajasthan Royals': {
        'name': 'Rajasthan Royals',
        'value': [0, 0, 0, 0, 0, 1, 0, 0]
    },
    'Royal Challengers Bangalore': {
        'name': 'Royal Challengers Bangalore',
        'value': [0, 0, 0, 0, 0, 0, 1, 0]
    },
    'Sunrisers Hyderabad': {
        'name': 'Sunrisers Hyderabad',
        'value': [0, 0, 0, 0, 0, 0, 0, 1]
    },
}

@app.route('/')
def home():
    return render_template('index.html', teams=TEAMS)

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = []
    if request.method == 'POST':
        batting_team = request.form['batting-team']

        temp_array += TEAMS[batting_team]['value']

        bowling_team = request.form['bowling-team']

        temp_array += TEAMS[bowling_team]['value']

        overs = float(request.form['overs'])
        score = int(request.form['score'])
        wickets = int(request.form['wickets'])
        score_before_5_overs = int(request.form['score_before_5_overs'])
        wickets_down_before_5_overs = int(request.form['wickets_down_before_5_overs'])

        temp_array += [overs, score, wickets, score - score_before_5_overs, wickets - wickets_down_before_5_overs]
        data = np.array([temp_array])
        regressor = load('data/model.joblib')
        my_prediction = int(regressor.predict(data)[0])

        context = {
            'lower_limit': my_prediction - 5,
            'upper_limit': my_prediction + 10
        }

        return render_template('result.html', **context)


if __name__ == '__main__':
    FLASK_ENV = environ.get('FLASK_ENV')
    IS_DEV = FLASK_ENV == 'development'
    app.run(debug=IS_DEV)