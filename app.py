from flask import Flask, request, render_template, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange, ValidationError
import pickle
import pandas as pd
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model
pipe = pickle.load(open('SVM.pkl', 'rb'))

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bengaluru', 
         'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings', 
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

class MatchPredictionForm(FlaskForm):
    batting_team = SelectField('Select the Batting Team', choices=[(team, team) for team in sorted(teams)], validators=[DataRequired()])
    bowling_team = SelectField('Select the Bowling Team', choices=[(team, team) for team in sorted(teams)], validators=[DataRequired()])
    selected_city = SelectField('Home ground', choices=[(city, city) for city in sorted(cities)], validators=[DataRequired()])
    target = IntegerField('Target', validators=[DataRequired(), NumberRange(min=0, max=799, message="Target should be between 0 and 799.")])
    score = IntegerField('Score', validators=[DataRequired(), NumberRange(min=0, message="Score should be a positive number.")])
    overs = FloatField('Overs Completed', validators=[DataRequired()])
    wickets = IntegerField('Wickets Out', validators=[DataRequired(), NumberRange(min=0, max=9, message="Wickets should be between 0 and 9.")])
    submit = SubmitField('Predict')

    def validate_score(self, field):
        if field.data > self.target.data:
            raise ValidationError("Score cannot exceed the target.")

    def validate_overs(self, field):
        overs_pattern = r'^([0-9]|1[0-9]|20)(\.[0-5])?$'
        if not re.match(overs_pattern, str(field.data)):
            raise ValidationError("Overs should be in the format of X.Y where X is the over number (less than or equal to 20) and Y is the ball number (0 to 5).")

@app.route('/')
def home():
    form = MatchPredictionForm()
    return render_template('index.html', form=form)

@app.route('/home')
def home1():
    form = MatchPredictionForm()
    return render_template('index.html', form=form)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = MatchPredictionForm()
    if form.validate_on_submit():
        batting_team = form.batting_team.data
        bowling_team = form.bowling_team.data
        selected_city = form.selected_city.data
        target = form.target.data
        score = form.score.data
        overs = form.overs.data
        wickets = form.wickets.data

        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        prediction = pipe.predict_proba(input_data)[0][1]
        win = prediction
        loss = 1 - prediction

        return render_template('index.html', form=form, win=win, loss=loss)
    
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')










