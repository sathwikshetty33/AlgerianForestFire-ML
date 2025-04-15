from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn .linear_model import LinearRegression

application = Flask(__name__)
app=application

ridge_model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        try:
            fields = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "ISI", "Classes", "Region"]
            data = [float(request.form[field]) for field in fields]
            scaler_data = scaler.transform([data])
            prediction = ridge_model.predict(scaler_data)
            prediction = prediction[0]
            return render_template('index.html', result=prediction)
        except ValueError:
            return render_template('index.html', error="Invalid input. Please enter numeric values.")
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)