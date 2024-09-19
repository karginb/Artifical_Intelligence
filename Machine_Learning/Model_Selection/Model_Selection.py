import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
#%%
iris = load_iris()

x = iris.data
y = iris.target
#%%
x = (x-np.min(x) / (np.max(x) - np.min(x)))
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
#%% knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
#%% K fol CV k = 10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=knn, X = x_train,y = y_train,cv = 10)
print("average accuracy:",np.mean(accuracies))
print("average std:",np.std(accuracies))
#%%
knn.fit(x_train, y_train)
print("test accuracy:",knn.score(x_test, y_test))
#%% grid search cross validation

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x_train,y_train)
#%%
print("tune hyperparameter K:",knn_cv.best_params_)
print("best accuracy by tuned hyperparameter:",knn_cv.best_score_)
#%% Grid Search Cross Validation with Logistic Regression
x_log = x[:100,:]
y_log = y[:100]
x_log_train,x_log_test,y_log_train,y_log_test = train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x_log_train,y_log_train)
#%%
print("tune hyperparameter K:",logreg_cv.best_params_)
print("best accuracy by tuned hyperparameter:",logreg_cv.best_score_)
