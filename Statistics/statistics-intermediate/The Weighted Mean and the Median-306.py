## 1. Introduction ##

mean_new = houses_per_year['Mean Price'].mean()
mean_original = houses['SalePrice'].mean()

difference = mean_original - mean_new

## 2. Different Weights ##

total_purchase = 0
for year in houses_per_year:
    total_purchase += houses_per_year['Mean Price'] * houses_per_year['Houses Sold']
    
    
weighted_mean = total_purchase/sum(houses_per_year['Houses Sold'])
mean_original = houses['SalePrice'].mean()

difference = round(weighted_mean, 10) - round(mean_original, 10)

## 3. The Weighted Mean ##

import numpy as np
def weighted_mean(mean_array, weights):
    num = 0
    den = 0
    for i in range(len(mean_array)):
        num += mean_array[i] * weights[i]
        den += weights[i]
    return num / den

weighted_mean_function = weighted_mean(houses_per_year['Mean Price'], houses_per_year['Houses Sold'])
weighted_mean_numpy = np.average(houses_per_year['Mean Price'], weights = houses_per_year['Houses Sold'])
                                                                                                                                                        
equal = round(weighted_mean_function, 10) == round(weighted_mean_numpy, 10)                                                                                        

## 4. The Median for Open-ended Distributions ##

distribution1 = [23, 24, 22, '20 years or lower,', 23, 42, 35]
distribution2 = [55, 38, 123, 40, 71]
distribution3 = [45, 22, 7, '5 books or lower', 32, 65, '100 books or more']

median1 = 23
median2 = 55
median3 = 32

## 5. Distributions with Even Number of Values ##

rooms_abv = houses['TotRms AbvGrd'].replace('10 or more', 10).copy()

rooms_sorted = rooms_abv.astype('int').sort_values()

middle_values = int(len(rooms_sorted)/2) 
if len(rooms_sorted) % 2 != 0:
    median = rooms_sorted.iloc[middle_values]
else:
    median = (rooms_sorted.iloc[middle_values] + rooms_sorted.iloc[middle_values+1])/2

## 6. The Median as a Resistant Statistic ##

import matplotlib.pyplot as plt

houses[['Lot Area', 'SalePrice']].plot.box()
plt.show()

lotarea_difference = houses['Lot Area'].mean() - houses['Lot Area'].median()
saleprice_difference = houses['SalePrice'].mean() - houses['SalePrice'].median()


## 7. The Median for Ordinal Scales ##

mean = houses['Overall Cond'].mean()
median = houses['Overall Cond'].median()

houses['Overall Cond'].plot.hist()
more_representative = 'mean'