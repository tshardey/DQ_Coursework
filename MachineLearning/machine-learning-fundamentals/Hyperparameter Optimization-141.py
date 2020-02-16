## 1. Recap ##

import pandas as pd

train_df = pd.read_csv("dc_airbnb_train.csv")
test_df = pd.read_csv("dc_airbnb_test.csv")

## 2. Hyperparameter optimization ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

hyper_params = list(range(1,6))
features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']

mse_values = []

for param in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=param, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse_values.append(mean_squared_error(test_df['price'], predictions))
    
print(mse_values)
    

## 3. Expanding grid search ##

hyper_params = list(range(1,21))
features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']

mse_values = []

for param in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=param, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse_values.append(mean_squared_error(test_df['price'], predictions))

print('mse_values')
                  

## 4. Visualizing hyperparameter values ##

import matplotlib.pyplot as plt
%matplotlib inline

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
hyper_params = [x for x in range(1, 21)]
mse_values = list()

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)
    
plt.scatter(hyper_params, mse_values)
plt.show()

## 5. Varying features and hyperparameters ##

hyper_params = [x for x in range(1,21)]
mse_values = list()

train_target = train_df['price']
train_feat = train_df.drop('price', axis=1)

test_target = test_df['price']
test_feat = test_df.drop('price', axis=1)

for param in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = param, algorithm='brute')
    knn.fit(train_feat, train_target)
    predictions = knn.predict(test_feat)
    mse_values.append(mean_squared_error(test_target, predictions))
    
plt.scatter(hyper_params, mse_values)
plt.show()

## 6. Practice the workflow ##

two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1,21)]
# Append the first model's MSE values to this list.
two_mse_values = list()

for param in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=param, algorithm='brute')
    knn.fit(train_df[two_features], train_df['price'])
    predictions = knn.predict(test_df[two_features])
    two_mse_values.append([param, mean_squared_error(test_df['price'], predictions)])
            
            
# Append the second model's MSE values to this list.
three_mse_values = list()

for param in hyper_params:
    knn2 = KNeighborsRegressor(n_neighbors=param, algorithm='brute')
    knn2.fit(train_df[three_features], train_df['price'])
    predictions2 = knn2.predict(test_df[three_features])
    three_mse_values.append([param, mean_squared_error(test_df['price'], predictions2)])

two_key = 0
two_value = 1000000
three_key = 0
three_value = 1000000

for item in two_mse_values:
    if item[1] < two_value:
        two_value = item[1]
        two_key = item[0]
        
two_hyp_mse = {two_key: two_value}

for item in three_mse_values:
    if item[1] < three_value:
        three_value = item[1]
        three_key = item[0]
        
three_hyp_mse = {three_key: three_value}