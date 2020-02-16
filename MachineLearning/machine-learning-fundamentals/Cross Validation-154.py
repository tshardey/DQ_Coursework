## 1. Introduction ##

import numpy as np
import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

dc_listings = dc_listings.iloc[np.random.permutation(len(dc_listings))]

split_one = dc_listings.iloc[:1862].copy()
split_two = dc_listings.iloc[1862:].copy()

## 2. Holdout Validation ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one

knn = KNeighborsRegressor()
knn.fit(train_one[['accommodates']], train_one['price'])
predicted_price = knn.predict(test_one[['accommodates']])
iteration_one_rmse = np.sqrt(mean_squared_error(test_one['price'], predicted_price))

knn2 = KNeighborsRegressor()
knn2.fit(train_two[['accommodates']], train_two['price'])
predicted_price2 = knn2.predict(test_two[['accommodates']])
iteration_two_rmse = np.sqrt(mean_squared_error(test_two['price'], predicted_price2))

avg_rmse = np.mean([iteration_one_rmse, iteration_two_rmse])

## 3. K-Fold Cross Validation ##

dc_listings.loc[dc_listings.index[0:745], "fold"] = 1
dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2
dc_listings.loc[dc_listings.index[1490:2234], "fold"] = 3
dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
dc_listings.loc[dc_listings.index[2978:3723], "fold"] = 5

print(dc_listings['fold'].value_counts())
print("\n Num of missing values: ", dc_listings['fold'].isnull().sum())

## 4. First iteration ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

test_iteration_one = dc_listings[dc_listings["fold"] == 1]
train_iteration_one = dc_listings[dc_listings["fold"] != 1]

knn = KNeighborsRegressor()
knn.fit(train_iteration_one[['accommodates']], train_iteration_one['price'])
labels = knn.predict(test_iteration_one[['accommodates']])
iteration_one_rmse = np.sqrt(mean_squared_error(test_iteration_one['price'], labels))

        

## 5. Function for training models ##

# Use np.mean to calculate the mean.
import numpy as np
fold_ids = [1,2,3,4,5]

def train_and_validate(df, folds):
    rmses = []
    for fold in folds:
        Train = df[df["fold"] != fold]
        Test = df[df["fold"] == fold]
        knn = KNeighborsRegressor()
        knn.fit(Train[['accommodates']], Train['price'])
        Predict = knn.predict(Test[['accommodates']])
        rmses.append(np.sqrt(mean_squared_error(Test['price'], Predict)))
    return rmses
                   
        
rmses = train_and_validate(dc_listings, fold_ids)
avg_rmse = np.average(rmses)

print(rmses, avg_rmse)

## 6. Performing K-Fold Cross Validation Using Scikit-Learn ##

from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 1)

knn = KNeighborsRegressor()
X = dc_listings[['accommodates']]
y = dc_listings['price']
mses = cross_val_score(knn, X, y, scoring='neg_mean_squared_error', cv=kf)
avg_rmse = np.mean(np.sqrt(np.absolute(mses)))

## 7. Exploring Different K Values ##

from sklearn.model_selection import cross_val_score, KFold

num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]

for fold in num_folds:
    kf = KFold(fold, shuffle=True, random_state=1)
    model = KNeighborsRegressor()
    mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
    rmses = np.sqrt(np.absolute(mses))
    avg_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(str(fold), "folds: ", "avg RMSE: ", str(avg_rmse), "std RMSE: ", str(std_rmse))