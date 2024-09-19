import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lineer_regression_dataset.csv",sep=";")

plt.scatter(df.experience,df.salary)
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()
#%% linear regression

#sklearn library
from sklearn.linear_model import LinearRegression
#linear regression model
linear_reg = LinearRegression()

x = df.experience.values.reshape(-1,1)
y = df.salary.values.reshape(-1,1)

linear_reg.fit(x,y)
#%% prediction
import numpy as np
b0 = linear_reg.predict([[0]])
print("b0:",b0)

b0_ = linear_reg.intercept_

b1 = linear_reg.coef_
print("b1:",b1)

new_salary = 1416 + 1162*11
print(new_salary)

print(linear_reg.predict([[11]]))

#visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)

plt.plot(array,y_head,color="red")
plt.show()
