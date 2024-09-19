import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%read
data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]
print(data.info)

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)
#%%normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#(x-min(x))/(max(x)-min(x))
#%%train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train:",x_train.shape)
#%% parameter initialization and sigmoid function
#dimension = 30
def initialization_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b


def sigmoid_fuction(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
#%% forward,backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid_fuction(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = np.sum(loss)/x.train.shape[1]
    #backward propragation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weights":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,gradients
#%% uptading(learning) parameters
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    #uptading(learning) parameters is number_of_iteration
    for i in range(number_of_iteration):
        #make forward and backward propagation and find and gradients
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        
        
        w = w - learning_rate * gradients["derivative_weights"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(c)
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
    
    parameters = {"weight":weight,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation="vertical")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters,gradients,cost_list

def predict(w,b,x_test):
    #x_test is input for forward propagation
    z = sigmoid_fuction(np.dot(w.T,x_train)+b)
    y_prediction = np.zeros(1,x_test.shape[1])
    #if z is bigger than 0.5,our prediction is sign one (y_head=1)
    #if z is smaller than 0.5,our prediction is sign zero(y_head=0)
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction
#%% logistic regression
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iteration):
    dimension = x_train.shape[0]
    w,b = initialization_weights_and_bias(dimension)
    #do not change learning rate
    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,num_iteration)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    print("test accuracy: {} %".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))


logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iteration=50)
#%% logistic regression with sklearn
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    