## 1. The Range ##

import pandas as pd
houses = pd.read_table('AmesHousing_1.txt')

def range_func(array):
    return max(array) - min(array)

range_by_year = {}
for year in houses["Yr Sold"].unique():
    houses_by_year = houses[houses["Yr Sold"] == year]
    range_by_year[year] = range_func(houses_by_year["SalePrice"])

    
one = False
two = True

## 2. The Average Distance ##

import numpy as np
C = [1,1,1,1,1,1,1,1,1,21]

def avg_distance(array):
    array_mean = np.mean(array)
    distances = []
    for value in array:
        distances.append(value - array_mean)
    return np.mean(distances)

avg_distance = avg_distance(C)

## 3. Mean Absolute Deviation ##

import numpy as np
C = [1,1,1,1,1,1,1,1,1,21]

def mean_absolute_deviation(array):
    array_mean = np.mean(array)
    absolute_distance = []
    for value in array:
        absolute_distance.append(abs(value - array_mean))
    return np.mean(absolute_distance)

mad = mean_absolute_deviation(C)

## 4. Variance ##

import numpy as np
C = [1,1,1,1,1,1,1,1,1,21]

def squared_distance(array):
    distances = []
    mean = np.mean(array)
    for value in array:
        distances.append((value-mean)**2)
    return sum(distances)/len(array)

variance_C = squared_distance(C)

## 5. Standard Deviation ##

from math import sqrt
from numpy import mean 

C = [1,1,1,1,1,1,1,1,1,21]

def standard_deviation(array):
    distances = []
    array_mean = mean(array)
    for value in array:
        distances.append((value-array_mean)**2)
    return sqrt(sum(distances)/len(array))

standard_deviation_C = standard_deviation(C)
    

## 6. Average Variability Around the Mean ##

def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
        
    variance = sum(distances) / len(distances)
    
    return sqrt(variance)

variability = {}
for year in houses['Yr Sold'].unique():
    houses_year = houses[houses['Yr Sold'] == year]
    variability[year] = standard_deviation(houses_year['SalePrice'])

greatest_variability = max(variability, key = variability.get)
lowest_variability= min(variability, key = variability.get)

## 7. A Measure of Spread ##

sample1 = houses['Year Built'].sample(50, random_state = 1)
sample2 = houses['Year Built'].sample(50, random_state = 2)

def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / len(distances)
    
    return sqrt(variance)

bigger_spread = 'sample 2'

st_dev1 = standard_deviation(sample1)
st_dev2 = standard_deviation(sample2)

## 8. The Sample Standard Deviation ##

def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / len(distances)
    
    return sqrt(variance)

import matplotlib.pyplot as plt

sale_samples = []

for i in range(5000):
    sampled = houses['SalePrice'].sample(10, random_state=i)
    sale_samples.append(standard_deviation(sampled))

plt.hist(sale_samples)
plt.axvline(standard_deviation(houses['SalePrice']), color='green', label='Population Standard Deviation')
plt.show()
    
    

## 9. Bessel's Correction ##

def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / (len(distances)-1)
    
    return sqrt(variance)

import matplotlib.pyplot as plt
st_devs = []

for i in range(5000):
    sample = houses['SalePrice'].sample(10, random_state = i)
    st_dev = standard_deviation(sample)
    st_devs.append(st_dev)
    
plt.hist(st_devs)
plt.axvline(standard_deviation(houses['SalePrice']))

## 10. Standard Notation ##

sample = houses.sample(100, random_state = 1)
from numpy import std, var

numpy_stdev = std(sample, ddof=1)
pandas_stdev=sample.std(ddof=1)

equal_stdevs = pandas_stdev == numpy_stdev

pandas_var = sample.var(ddof=1)
numpy_var = var(sample, ddof=1)
equal_vars = pandas_var == numpy_var
                                 

## 11. Sample Variance — Unbiased Estimator ##

from numpy import std, var, mean
population = [0, 3, 6]

samples = [[0,3], [0,6],
           [3,0], [3,6],
           [6,0], [6,3]
          ]

sample_var = []
sample_mean = []

for pair in samples:
    sample_var.append(var(pair))
    sample_mean.append(std(pair))
    
equal_var = mean(sample_var) == mean(population)
equal_stdev = mean(sample_mean) == std(population)
