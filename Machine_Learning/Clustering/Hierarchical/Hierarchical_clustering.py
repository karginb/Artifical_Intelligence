import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#%%
x1 = np.random.normal(25,5,100)
y1 = np.random.normal(25,5,100)

x2 = np.random.normal(55,5,100)
y2 = np.random.normal(60,5,100)


x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)

dictionary = {"x":x,"y":y}
df = pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()
#%%
from scipy.cluster.hierarchy import linkage,dendrogram
merg = linkage(df,method="ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("Euclidean distance")
plt.show()
#%% HC
from sklearn.cluster import AgglomerativeClustering
hierarchical_cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
clusters = hierarchical_cluster.fit_predict(df)
df["label"] = clusters
plt.scatter(df.x[df.label == 0],df.y[df.label == 0],color = "red")
plt.scatter(df.x[df.label == 1],df.y[df.label == 1],color = "green")
plt.scatter(df.x[df.label == 2],df.y[df.label == 2],color = "blue")
plt.show()