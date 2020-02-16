## 2. Introduction To The Data ##

import pandas as pd

data = pd.read_table("AmesHousing.txt", delimiter='\t')

train = data[:1460].copy()
test = data[1460:].copy()

target = 'SalePrice'
data.info()

## 3. Simple Linear Regression ##

import matplotlib.pyplot as plt
# For prettier plots.
import seaborn

plt.scatter(data['Garage Area'], data['SalePrice'])
plt.show()
plt.scatter(data['Gr Liv Area'], data['SalePrice'])
plt.show()
plt.scatter(data['Overall Cond'], data['SalePrice'])
plt.show()

## 5. Using Scikit-Learn To Train And Predict ##

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train[['Gr Liv Area']], train['SalePrice'])

a1 = reg.coef_
a0 = reg.intercept_

## 6. Making Predictions ##

import numpy as np
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
predict_train = lr.predict(train[['Gr Liv Area']])
predict_test = lr.predict(test[['Gr Liv Area']])

train_rmse = np.sqrt(mean_squared_error(train['SalePrice'], predict_train))
test_rmse = np.sqrt(mean_squared_error(test['SalePrice'], predict_test))
                     



## 7. Multiple Linear Regression ##

cols = ['Overall Cond', 'Gr Liv Area']

lr = LinearRegression()
lr.fit(train[cols], train['SalePrice'])

train_rmse_2 = np.sqrt(mean_squared_error(train['SalePrice'], lr.predict(train[cols])))

test_rmse_2 = np.sqrt(mean_squared_error(test['SalePrice'], lr.predict(test[cols])))