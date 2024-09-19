import pandas as pd
import numpy as np 
from sklearn.datasets import load_iris
#%%
iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns=feature_names)
df["class"] = y

x = data
#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=2,whiten=True)
pca.fit(x)
x_pca = pca.transform(x)

print("variance ratio:",pca.explained_variance_ratio_)
print("sum:",sum(pca.explained_variance_ratio_))
#%% 2D visualize
import matplotlib.pyplot as plt
df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

for i in range(3):
    plt.scatter(df.p1[df["class"] == i],df.p2[df["class"] == i],color = color[i],label = iris.target_names[i])
plt.xlabel("p1")
plt.ylabel("p2")
plt.legend()
plt.show()