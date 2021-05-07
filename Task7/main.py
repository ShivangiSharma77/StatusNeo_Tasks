#-----------flask-------

import numpy as np
from flask import Flask, request
from tensorflow import keras
#from flask_ngrok import run_with_ngrok
# from tensorflow.keras.models import load_model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import load_model


m = None
app = Flask(__name__)
#run_with_ngrok(app)


def load_model_func():
    global m
    # model variable refers to the global variable
    m = keras.models.load_model('model_files/b_model.h5')

def fit_scaler():
    global dframe
    dframe = pd.read_csv("model_files/coin_Bitcoin (1).csv")
    global scaler1
    scaler1 = MinMaxScaler(feature_range=(0,1))
    global price1
    price1 = dframe.Close.values.reshape(-1, 1)
    price1 = price1.astype('float32')
    scaler1.fit(price1)
    
    
@app.route('/')
def home_endpoint():
    return 'Welcome!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    load_model_func() 
    fit_scaler()
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)
        
        transform_d = scaler1.transform(data)
        ar = []
        for index in range(len(transform_d)+1 - len(transform_d)):
            ar.append(transform_d[index: index + len(transform_d)])
        
        prediction = m.predict(np.array(ar))  # runs globally loaded model on the data
        res=scaler1.inverse_transform(prediction[0].reshape(-1,1))
    return str(res)


# if __name__ == '__main__':
#     # load_model_func()  # load model at the beginning once only
#     app.run(host='localhost', port=5000)

#    app.run()