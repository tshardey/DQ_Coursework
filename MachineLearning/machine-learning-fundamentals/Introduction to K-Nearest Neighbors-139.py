## 2. Introduction to the data ##

import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")

print(dc_listings.head(1))

## 4. Euclidean distance ##

import numpy as np

our_acc_value = 3

first_value = dc_listings['accommodates'][0]

first_distance = np.abs(our_acc_value - first_value)

## 5. Calculate distance for all observations ##

import numpy as np
our_dc_listing = 3
dc_listings['distance'] = dc_listings["accommodates"].apply([lambda x: np.abs(x-our_dc_listing)])

print(dc_listings['distance'].value_counts())

## 6. Randomizing, and sorting ##

import numpy as np
np.random.seed(1)

index_array = np.random.permutation(len(dc_listings))

dc_listings = dc_listings.loc[index_array]
dc_listings = dc_listings.sort_values('distance')
print(dc_listings['price'][:10])

## 7. Average price ##

replaced = [",","$"]
stripped = dc_listings['price']
for item in replaced:
    stripped = stripped.str.replace(item, "")
    
dc_listings['price'] = stripped.astype(float)

mean_price = dc_listings['price'][0:5].mean()

print(mean_price)

## 8. Function to make predictions ##

import numpy as np
# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def predict_price(new_listing):
    temp_df = dc_listings.copy()
    ## Complete the function.
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    return np.mean(temp_df['price'][0:5])

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)