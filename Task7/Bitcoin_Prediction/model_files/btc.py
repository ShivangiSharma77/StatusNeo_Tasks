from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df2 = pd.read_csv("coin_bitcoin (1).csv")

df2 = df2.sort_values('Date')
scaler = MinMaxScaler(feature_range=(0,1))
price = df2.Close.values.reshape(-1, 1)
price = price.astype('float32')
scaled_price = scaler.fit_transform(price)

len_train=int(len(scaled_price)*0.8)
len_test=len(scaled_price)-len_train
train,test=scaled_price[0:len_train,:],scaled_price[len_train:len(scaled_price),:]
#print(len(train),len(test))
#print(len_train,len_test)

def getsequences(data, sequence_len):
    d = []
    for index in range(len(data) - sequence_len):
        d.append(data[index: index + sequence_len])

    return np.array(d)

def getprocessedvalues(currdata, sequence_len):
    data = getsequences(currdata, sequence_len)
  
    xtrain = data[:len_train, :-1, :]
    ytrain = data[:len_train, -1, :]

    xtest = data[len_train:, :-1, :]
    ytest = data[len_train:, -1, :]

    return xtrain, ytrain, xtest, ytest

xtrain, ytrain, xtest, ytest =getprocessedvalues(scaled_price, 20)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation  

model = models.Sequential()
model.add(LSTM(30, input_shape=(19,1)))  #input_shape=(length of sequence,num of features)
model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

hist=model.fit(xtrain,ytrain,epochs=100,batch_size=30, validation_data=(xtest, ytest))
model.evaluate(xtest,ytest)

from tensorflow.keras.models import load_model

model.save('b_model.h5')  # creates a HDF5 file 'my_model.h5'