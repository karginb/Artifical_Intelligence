import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#%%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values
#%%
x = (x_data - np.min(x_data) / (np.max(x_data) - np.min(x_data)))
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#%%
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("accuracy of nb algorithm: ",nb.score(x_test,y_test))