import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_linear_regression.csv",sep=";")
x = df.cars_price.values.reshape(-1,1)
y = df.car_max_velocity.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("car_max_velocity")
plt.xlabel("cars_prices")
#%% linear regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x, y)
#%%prediction
y_head = linear_regression.predict(x)

plt.plot(x,y_head,color="red")
plt.show()

print("the car which is price is 10 million:" + linear_regression.predict([[10000]]))
#%%polynomial regression
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=2)

x_polynomial = polynomial_regression.fit_transform(x)
#%% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial, y)
#%%

y_head2 = linear_regression2.predict(x_polynomial)
plt.plot(x,y_head2,color = "green",label="polynomial")
plt.legend()
plt.show()
