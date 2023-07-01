#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[ ]:


start = '2010-01-01'
end = '2022-07-29'

df = data.DataReader('AAPL', 'yahoo', start, end)
df.tail()


# In[ ]:


df = df.reset_index()
df.head()


# In[ ]:


plt.plot(df.Close)


# In[ ]:


df


# 

# In[ ]:


ma100 = df.Close.rolling(100).mean()
ma100


# In[ ]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')


# In[ ]:


ma200 = df.Close.rolling(200).mean()
ma200


# In[ ]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# In[ ]:


df.shape


# In[ ]:


#SPLITTING DATA INTO TRAINING AND TESTING

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)


# In[ ]:


data_training


# In[ ]:


data_testing


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[ ]:


data_training_array = scaler.fit_transform(data_training)
data_training_array.shape


# In[ ]:


from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[ ]:


x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
    
    
x_train, y_train = np.array(x_train), np.array(y_train)


# In[ ]:


model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
              input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))


# In[ ]:


x_train.shape


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)


# In[ ]:


model.save('keras_model.h5')


# In[ ]:


data_testing.head()


# In[ ]:


data_training.tail(100)


# In[ ]:


past_100_days = data_training.tail(100)


# In[ ]:


final_df = past_100_days.append(data_testing, ignore_index=True)


# In[ ]:


final_df.head()


# In[ ]:


input_data = scaler.fit_transform(final_df)
input_data


# In[ ]:


input_data.shape


# In[ ]:


x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])


# In[ ]:


x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


#Making Predictions

y_predicted = model.predict(x_test)


# In[ ]:


y_predicted.shape


# In[ ]:


y_test


# In[ ]:


y_predicted


# In[ ]:


scaler.scale_


# In[ ]:


scale_factor = 1/0.00690691
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# In[ ]:


plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Prices')
plt.plot(y_predicted, 'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


start = '2010-01-01'
end = '2023-07-28'


st.title('Stock Trends Predictor')
user_input = st.text_input('Enter Stock Ticker','AAPL' )
df = data.DataReader(user_input, 'yahoo', start, end)

#Describe Data

st.subheader('Data from 2010 to 2022')
st.write(df.describe())

st.subheader('Closing Time vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Time vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Time vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)




#Load Model
file_name = os.path.dirname(__file__) +'//keras_model.h5'
model = load_model('keras_model.h5')

#Testing

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12, 6))

plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Prices')
plt.plot(y_predicted, 'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




