import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("random_forest_regression_dataset.csv",sep=";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
#%% random forest regression
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators=100,random_state=42)
random_forest.fit(x, y)

random_forest.predict([[7.8]]) 
x_new = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = random_forest.predict(x_new)
#%% visualization
plt.scatter(x,y,color = "red")
plt.plot(x_new,y_head,color="green")
plt.xlabel("tribune level")
plt.ylabel("price")
plt.show()