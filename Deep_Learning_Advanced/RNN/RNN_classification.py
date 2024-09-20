import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
#%%
dataset_train = pd.read_csv("Stock_Price_Train.csv")
dataset_train.head()
train = dataset_train.loc[:,["Open"]].values
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train)
train_scaled
#%%
X_train = []
Y_train = []
timesteps = 50
for i in range(timesteps,1258):
    X_train.append(train_scaled[i - timesteps:i,0])
    Y_train.append(train_scaled[i,0])
X_train,Y_train = np.array(X_train),np.array(Y_train)
X_train
#%%
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_train
#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SimpleRNN

regressor = Sequential()

regressor.add(SimpleRNN(units = 50,activation = "tanh",return_sequences = True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50,activation = "tanh",return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50,activation = "tanh",return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = "adam",loss = "mean_squared_error")

regressor.fit(X_train,Y_train,epochs = 100,batch_size = 32)

#%%
dataset_test = pd.read_csv("Stock_price_Test.csv")
dataset_test.head()
real_stock_price = dataset_test.loc[:,["Open"]].values
real_stock_price
#%%
dataset_total = pd.concat(dataset_train["Open"],dataset_test["Open"],axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)
inputs
#%%
X_test = []
for i in range(timesteps,70):
    X_test.append(inputs[i - timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
preedicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
#%%
plt.plot(real_stock_price,color = "red",label = "Real Stock Price")
plt.plot(predicrted_stock_price,color = "blue",label = "Predicted Stock Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
#%%
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
#%%
data = pd.read_csv("international-airline-passengers.csv",skipfooter=5)
data.head()
#%%
dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("time")
plt.ylabel("Number of passengers")
plt.title("International Airline Passengers")
plt.show()
#%%
dataset = dataset.reshape(-1,1)
dataset = dataset.astype("float32")
dataset.shape
#%%
scaler = MinMaxScaler(feature_range = (0,1))
dataset = scaler.fit_transform(dataset)
#%%
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("train size , test size",len(train),len(test))
#%%
time_stemp = 10 
dataX = []
dataY = []
for i in range(len(train) - time_stemp - 1):
    a = train[i:(i+time_stemp),0]
    dataX.append(a)
    dataY.append(train[i + time_stemp,0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY)
#%%
dataX = []
dataY = []
for i in range(len(test) - time_stemp - 1):
    a = test[i:(i + time_stemp),0]
    dataX.append(a)
    dataY.append(test[i + time_stemp,0])
testX = np.array(dataX)
testY = np.array(dataY)
#%%
trainX = numpy.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = numpy.reshape(testX,(testX.shape[0],1,testX.shape[1]))
#%%
model = Sequential()
model.add(LSTM(10,input_shape=(1,time_stemp)))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer = "adam")
model.fit(trainX,trainY,epochs = 50,batch_size = 1)
#%%
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print("Train Score:" % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print("Test Score:" % (testScore))
#%%
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,:] = numpy.nan
trainPredictPlot[time_stemp:len(trainPredict) + time_stemp,:] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,:] = numpy.nan
testPredictPlot[time_stemp:len(testPredict) + (time_stemp * 2) + 1:len(dataset) - 1,:] = testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


#%%


















































