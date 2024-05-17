import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
## Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    # data = [float(x) for x in request.form.values()]
    # final_input = scalar.transform(np.array(data).reshape(1, -1))
    # print(final_input)
    # output = regmodel.predict(final_input)[0]
    # return render_template("home.html", prediction_text={}.format(output))
    
    age = request.form['Age']
    sex = request.form['Sex']
    cp = request.form['Cp']
    trestbps = request.form['Trestbps']
    chol = request.form['Chol']
    fbs = request.form['Fbs']
    restecg = request.form['Resttechg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    # Example prediction logic (replace with your model prediction)
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = [0]  # Replace with actual model prediction
    final_input = scalar.transform(features)
    prediction = regmodel.predict(final_input)
    prediction_text = 'The Person does not have a Heart Disease' if prediction[0] == 0 else 'The Person has Heart Disease'

    return render_template('home.html', prediction_text=prediction_text)




if __name__ == "__main__":
    app.run(debug=True)