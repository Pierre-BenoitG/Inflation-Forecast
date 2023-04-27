#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reset', '')


# In[ ]:


from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import plot_model 
from keras import models
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[ ]:


import csv 
import pandas as pd
import numpy as np
from numpy.random import seed


# In[ ]:


import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from datetime import timedelta
import time
import tensorflow as tf
import keras as keras
from keras.models import Sequential
import sklearn
from sklearn.model_selection import train_test_split


# In[ ]:


#Download the data which are a drive
from google.colab import drive
drive.mount('/content/drive')
serie_inflation = pd.read_csv('/content/drive/My Drive/Copie de CPIAUCSL.csv')
#we create a data frame with only the datas we will use for the model 
serie_inflation["DATE"]=pd.to_datetime(serie_inflation["DATE"])
serie_inflation.set_index('DATE',inplace=True)
serie_inflation


# In[ ]:


#Splitting data into test and train
splitting_date = 876 
train_data , validation_data = serie_inflation[:splitting_date] , serie_inflation[splitting_date:]


# In[ ]:


#Computing mean and standard deviation of train data
mean_train = train_data.mean()
std_train = train_data.std()

#Normalizing the data
train_data = (train_data - mean_train)/std_train
validation_data = (validation_data - mean_train)/std_train
serie_inflation = (serie_inflation - mean_train)/std_train

train_data


# In[ ]:


#define X_train and y_train using lagged data, we are using the inflation on the past 6 years in order to try to forecast the following data
# Hence they will be 72 lagged time-steps and 1 output

X_train  = []
y_train = []
lagg = 72 
for i in range(lagg,splitting_date):
  X_train.append(train_data[i-lagg:i])
  y_train.append(train_data[i:i+1])
X_train , y_train = np.array(X_train) , np.array(y_train)
print(X_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train.shape


# In[ ]:


network = models.Sequential()

#Adding the first LSTM layer 
network.add(LSTM(units = 100, activation = 'relu' ,  return_sequences = False, input_shape = (X_train.shape[1], 1)))
#Adding Classic layers
network.add(layers.Dense(50 , activation = 'relu'))
network.add(layers.Dense(50 , activation = 'relu'))
network.add(layers.Dense(15 , activation = 'relu'))

network.add(Dense(units = 1))

network.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), loss = 'mean_squared_error')

network.fit(X_train, y_train, epochs = 50)


# ##Short Term Forecasting

# In[ ]:


#Get the predicted values 

inputs = serie_inflation[len(serie_inflation) - len(validation_data) - lagg:]
X_test = []
for i in range(lagg , len(inputs)):
  X_test.append(inputs[i-lagg:i])
X_test = np.array(X_test)  

predicted_inflation = network.predict(X_test)


# In[ ]:


predicted_inflation


# In[ ]:


plt.plot(validation_data.index , predicted_inflation)
plt.plot(validation_data)


# In[ ]:


#Return to unnormalize prediction with the normalize one
forecasting_inflation = pd.DataFrame(index = validation_data.index , data = {
    "Normalize Forecasting": predicted_inflation.tolist(),
    "Forecasting": (predicted_inflation * std_train.values[0] + mean_train.values[0]).tolist()
})

for i in range(len(forecasting_inflation)):
  forecasting_inflation["Forecasting"][i] = np.array(forecasting_inflation["Forecasting"][i])[0]
  forecasting_inflation["Normalize Forecasting"][i] = np.array(forecasting_inflation["Normalize Forecasting"][i])[0]

forecasting_inflation


# In[ ]:


#Plot the forecast
plt.figure(figsize=(10,8))
plt.plot(validation_data *std_train.values[0] + mean_train.values[0]  , color = "blue" , label= "Observed Inflation")
plt.plot(forecasting_inflation["Forecasting"] , color = "green" , label = "Inflation Forecasted by the Model")
plt.ylabel("American CPI index normalize")
plt.xlabel("Date")
plt.legend()
plt.grid()


# In[ ]:


#Relative Error
Relative_Error = np.abs(forecasting_inflation["Forecasting"].values - (validation_data["CPIAUCSL"].values * std_train.values[0] + mean_train.values[0])) / (validation_data["CPIAUCSL"].values * std_train.values[0] + mean_train.values[0])
plt.figure(figsize=(15,7))
plt.plot(validation_data.index , Relative_Error * 100 , color = "blue" , label = "Relative Error")
plt.ylabel("Relative Error in percentage %")
plt.xlabel("Date")
plt.legend()
plt.grid()


# ##Long Term Forecasting

# In[ ]:


## Forecast values using the forecast values
## Creation the input for the each simulation


inputs = serie_inflation[len(serie_inflation) - len(validation_data) - lagg:]
dynamics_prediction = []
for i in range(len(validation_data)):
  X_test = []
  X_test.append(inputs[i:i+lagg])
  X_test = np.array(X_test)
  prediction = network.predict(X_test) #computing prediction

  inputs[i+lagg:i+1+lagg]["CPIAUCSL"].values[0] = prediction[0][0] #Change the known value of inflation by the predicted one
  dynamics_prediction.append(prediction[0][0])
  
display(dynamics_prediction)


# In[ ]:


#constructing the unnormalize index
dynamics_forecasting_inflation = pd.DataFrame(index = validation_data.index , data = {
    "Normalize Forecasting": dynamics_prediction,
    "Forecasting": (np.array(dynamics_prediction) * std_train.values[0] + mean_train.values[0]).tolist()
})
dynamics_forecasting_inflation


# In[ ]:


#Plot the forecast
plt.figure(figsize=(10,8))
plt.plot(validation_data *std_train.values[0] + mean_train.values[0]  , color = "blue" , label= "Observed Inflation")
plt.plot(dynamics_forecasting_inflation["Forecasting"] , color = "green" , label = "Inflation Forecasted by the Model")
plt.ylabel("American CPI index")
plt.xlabel("Date")
plt.legend()
plt.grid()


# In[ ]:


#Plot Relative Error
Relative_Error_dyn = np.abs(dynamics_forecasting_inflation["Forecasting"].values - (validation_data["CPIAUCSL"].values * std_train.values[0] + mean_train.values[0])) / (validation_data["CPIAUCSL"].values * std_train.values[0] + mean_train.values[0])
plt.figure(figsize=(10,8))
#plt.plot(validation_data.index , Relative_Error * 100 , color = "green" , label = "Relative Error of Model 1")
plt.plot(validation_data.index , Relative_Error_dyn * 100 , color = "blue" , label = "Relative Error of Model Inflation")
plt.ylabel("Relative Error in percentage % ")
plt.xlabel("Date")
plt.legend()
plt.grid()

