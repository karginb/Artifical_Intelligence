import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()
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
data.diagnosis =[1 if i == "M" else 0 for i in data.diagnosis]
x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values
#%% normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
#%% train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#%% knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#%% evaluate the knn model
print("{} knn score: {} ".format(3,knn.score(x_test, y_test)))
#%% find k value 
score_list = []
for i in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(x_train, y_train)
    score_list.append(knn.score(x_test,y_test))

plt.plot(range(1,15),score_list)
plt.xlabel("k value")
plt.ylabel("accuracy")
plt.show()