import pandas as pd
import helper
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

df = pd.read_csv('comp.csv')
df=df['Open'].values
df = df.reshape(-1,1)

dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[:int(df.shape[0]*0.8)])

scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

x_train,y_train = helper.create_dataset(dataset_train)
x_test,y_test = helper.create_dataset(dataset_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

model=Sequential()
model.add(LSTM(units=4,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=4))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train,y_train,epochs=5,batch_size=16,verbose=0)

predictions = model.predict(x_test)

model.summary()
print(predictions[:,-1])
