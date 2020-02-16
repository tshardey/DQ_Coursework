## 1. Introduction to the data ##

import pandas as pd
cars = pd.read_csv("auto.csv")

unique_regions = cars['origin'].unique()

print(unique_regions)

## 2. Dummy variables ##

dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)

dummy_years = pd.get_dummies(cars["year"], prefix="year")
cars = pd.concat([cars, dummy_years], axis=1)

cars.drop(['year', 'cylinders'], axis=1, inplace=True)

print(cars.head())


## 3. Multiclass classification ##

shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]

cut_off = int(0.7*shuffled_cars.shape[0])

train = shuffled_cars[:cut_off]
test = shuffled_cars[cut_off:]

## 4. Training a multiclass logistic regression model ##

from sklearn.linear_model import LogisticRegression

unique_origins = cars["origin"].unique()
unique_origins.sort()

cyl_columns = [col for col in train.columns if 'cyl' in col]
year_columns = [col for col in train.columns if 'year' in col]
keep_columns = cyl_columns + year_columns

models = {}

for origin in unique_origins:
    X = train[keep_columns]
    y = train['origin'] == origin
    
    lr = LogisticRegression()
    models[origin] = lr.fit(X, y)
        

## 5. Testing the models ##

testing_probs = pd.DataFrame(columns=unique_origins)

for origin, model in models.items():
    X_test= test[features]
    predictions = model.predict_proba(X_test)
    testing_probs[origin] = predictions[:,1]
    
    
testing_probs

## 6. Choose the origin ##

predicted_origins = testing_probs.idxmax(axis=1)

print(predicted_origins)