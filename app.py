import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Gn.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('./index.html', prediction_text='The suitable crop for this soil sample is {}'.format(final_prediction))


if __name__ == "__main__":
    app.run(debug=True)