import pandas as pd
import numpy as np 
import matplotplib.pyplot as plt
#%%
data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#%%
data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]
x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values
#%%
x = (x_data - np.min(x_data) - (np.max(x_data) - np.min(x_data)))
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)
#%%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("score: ",dt.score(x_test, y_test))