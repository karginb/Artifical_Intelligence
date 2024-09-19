import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

linear_df = pd.read_csv("linear_regression_dataset.csv",sep = ";")
polynomial_df = pd.read_csv("random_forest_regression_dataset.csv",sep=";",header=None)

x_polynomial = polynomial_df.iloc[:,0].values.reshape(-1,1)
y_polynomial = polynomial_df.iloc[:,1].values.reshape(-1,1)
#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x_polynomial, y_polynomial)
y_head = rf.predict(x_polynomial)
#%%
from sklearn.metrics import r2_score
print("r_score : ",r2_score(y_polynomial,y_head))
#%%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
x_linear = linear_df.experience.values.reshape(-1,1)
y_linear = linear_df.salary.values.reshape(-1,1)
lr.fit(x_linear, y_linear)
y_head_linear = lr.predict(x_linear)
#%%
print("r_score:",r2_score(y_linear,y_head_linear))