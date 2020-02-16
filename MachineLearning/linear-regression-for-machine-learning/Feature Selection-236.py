## 1. Missing Values ##

import pandas as pd
data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]
columns_to_drop = ['PID', 'Year Built', "Year Remod/Add", "Garage Yr Blt", "Mo Sold", "Yr Sold"]

numerical_train = train.select_dtypes(include=['int','float'])
numerical_train = numerical_train.drop(columns_to_drop, axis=1)

null_series = numerical_train.apply(lambda x: sum(x.isnull()))
full_cols_series = null_series[null_series == 0]





## 2. Correlating Feature Columns With Target Column ##

train_subset = train[full_cols_series.index]

train_corr = train_subset.corr()

sorted_corrs = train_corr.loc['SalePrice'].abs().sort_values()
print(sorted_corrs)

## 3. Correlation Matrix Heatmap ##

import seaborn as sns
import matplotlib.pyplot as plt

strong_corrs = sorted_corrs[sorted_corrs>0.3].index

sns.heatmap(train_corr.loc[strong_corrs, strong_corrs])

## 4. Train And Test Model ##

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

final_corr_cols = strong_corrs.drop(['Garage Cars', 'TotRms AbvGrd'])
features = final_corr_cols.drop(['SalePrice']).index
target = 'SalePrice'

test[final_corr_cols.index].info()

clean_test=test[final_corr_cols.index].dropna()

lr = LinearRegression().fit(train[features], train[target])
train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])
train_rmse = np.sqrt(mean_squared_error(train[target], train_predictions))
test_rmse = np.sqrt(mean_squared_error(clean_test[target], test_predictions))


## 5. Removing Low Variance Features ##

train[features] = (train[features]-train[features].min())/(train[features].max()-train[features].min())

train[features].describe()

sorted_vars = train[features].var().sort_values()

print(sorted_vars)


## 6. Final Model ##

features = features.drop('Open Porch SF')


lr = LinearRegression()
lr.fit(train[features], train[target])
train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])

train_rmse_2 = np.sqrt(mean_squared_error(train[target], train_predictions))
test_rmse_2 = np.sqrt(mean_squared_error(clean_test[target], test_predictions))

print(train_rmse_2, test_rmse_2)