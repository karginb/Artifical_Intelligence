import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
#%%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean,M.texture_mean,color="red",alpha=0.5,label="Melignant")
plt.scatter(B.radius_mean,B.texture_mean,color="green",alpha=0.5,label="Benign")
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.legend()
plt.show()
#%%
data.diagnosis[1 if i == "M" else 0 for i in data.diagnosis]
x_data = data.drop(["diagnosis"])
y = data.diagnosis.values
#%% normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
#%% train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#%% swm model
from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train,y_train)

print("accuracy of svm algorithm:",svm.score(x_test, y_test))