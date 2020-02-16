## 1. Individual Values ##

import pandas as pd
import matplotlib.pyplot as plt
houses = pd.read_table('AmesHousing_1.txt')

import matplotlib.pyplot as plt
houses['SalePrice'].plot.kde(xlim = (houses['SalePrice'].min(),
                                    houses['SalePrice'].max()
                                    )
                            )

st_dev = houses['SalePrice'].std(ddof = 0)
mean = houses['SalePrice'].mean()
plt.axvline(mean, color = 'Black', label = 'Mean')
plt.axvline(mean + st_dev, color = 'Red', label = 'Standard deviation')
plt.axvline(220000, color = 'Orange', label = '220000')
plt.legend()
                             

## 2. Number of Standard Deviations ##

from numpy import mean, std

distance = abs(220000-mean(houses['SalePrice']))

st_devs_away = distance/std(houses['SalePrice'])


## 3. Z-scores ##

from numpy import mean, std
min_val = houses['SalePrice'].min()
mean_val = houses['SalePrice'].mean()
max_val = houses['SalePrice'].max()


def z_score(value, array):
    array_mean = mean(array)
    array_std = std(array)
    return (value - array_mean)/array_std

min_z = z_score(min_val, houses['SalePrice'])
mean_z = z_score(mean_val, houses['SalePrice'])
max_z = z_score(max_val, houses['SalePrice'])

## 4. Locating Values in Different Distributions ##

def z_score(value, array, bessel = 0):
    mean = sum(array) / len(array)
    
    from numpy import std
    st_dev = std(array, ddof = bessel)
    
    distance = value - mean
    z = distance / st_dev
    
    return z

neighborhoods = ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst']

z_scores = {}
for neighborhood in neighborhoods:
    hood = houses[houses['Neighborhood']==neighborhood]
    z_scores[neighborhood] = z_score(220000, hood['SalePrice'])

best_investment = 'College Creek'

## 5. Transforming Distributions ##

mean = houses['SalePrice'].mean()
st_dev = houses['SalePrice'].std(ddof = 0)
houses['z_prices'] = houses['SalePrice'].apply(
    lambda x: ((x - mean) / st_dev)
    )

z_mean_price = houses['z_prices'].mean()
z_stdev_price = round(houses['z_prices'].std(),0)

area_mean = houses['Lot Area'].mean()
area_std = houses['Lot Area'].std(ddof = 0)

houses['area_z_prices'] = houses['Lot Area'].apply(
    lambda x: ((x-area_mean)/area_std))
z_mean_area = houses['area_z_prices'].mean()
z_stdev_area = round(houses['area_z_prices'].std(),0)

## 6. The Standard Distribution ##

from numpy import std, mean
population = [0,8,0,8]

def z_score(value, population):
    pop_mean = mean(population)
    pop_std = std(population)
    return (value-pop_mean)/pop_std

mean_z = z_score(mean(population), population)
stdev_z = z_score(mean(population) + std(population), population)

## 7. Standardizing Samples ##

from numpy import std, mean
sample = [0,8,0,8]

x_bar = mean(sample)
s = std(sample, ddof = 1)

standardized_sample = []
for value in sample:
    z = (value - x_bar) / s
    standardized_sample.append(z)
    
stdev_sample = std(standardized_sample, ddof=1)

## 8. Using Standardization for Comparisons ##

from numpy import mean, std

mean_1 = mean(houses['index_1'])
std_1 = std(houses['index_1'])
mean_2 = mean(houses['index_2'])
std_2 = std(houses['index_2'])

value_1 = houses['index_1'][1]
value_2 = houses['index_2'][0]

z_score_1 = (value_1-mean_1)/std_1
z_score_2 = (value_2-mean_2)/std_2

better = 'first'

## 9. Converting Back from Z-scores ##

from numpy import mean, std

houses['z_merged'] = houses['z_merged'].apply(lambda x: x*10+50)

mean_transformed = mean(houses['z_merged'])
stdev_transformed = std(houses['z_merged'])