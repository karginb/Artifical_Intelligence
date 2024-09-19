import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("decision_tree_regression_dataset.csv",sep=";",header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)
#%% decision tree regression
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(x, y)

decision_tree.predict([[5.5]])
x_new = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = decision_tree.predict(x_new)
#%% visualization
plt.scatter(x, y, color="green")
plt.plot(x_new,y_head,color="red")
plt.xlabel("tribune level")
plt.ylabel("price")
plt.show()