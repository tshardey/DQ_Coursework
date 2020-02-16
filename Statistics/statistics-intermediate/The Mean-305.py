## 2. The Mean ##

distribution = [0,2,3,3,3,4,13]

mean = sum(distribution)/len(distribution)
center = mean == (max(distribution)-min(distribution))/2

above_mean = []
below_mean = []

for i in distribution:
    if i > mean:
        above_mean.append(abs(mean-i))
    elif i < mean:
        below_mean.append(abs(mean-i))
 
equal_distances = sum(above_mean) == sum(below_mean)

## 3. The Mean as a Balance Point ##

from numpy.random import randint, seed

equal_distances = 0
for i in range(0, 5000):
    seed(i)
    random = randint(0, 1000, size=10)
    less_mean = []
    more_mean = []
    for value in random:
        if value > random.mean():
            more_mean.append(abs(value-random.mean()))
        elif value < random.mean():
            less_mean.append(abs(value-random.mean()))
    if round(sum(more_mean), 1) == round(sum(less_mean),1):
        equal_distances += 1
            

## 4. Defining the Mean Algebraically ##

one = False
two = False
three = False

## 5. An Alternative Definition ##

distribution_1 = [42, 24, 32, 11]
distribution_2 = [102, 32, 74, 15, 38, 45, 22]
distribution_3 = [3, 12, 7, 2, 15, 1, 21]

def long_mean(distribution): 
    N = len(distribution)
    distribution_sum = 0
    for i in range(N):
        distribution_sum += distribution[i]
    return distribution_sum/N

mean_1 = long_mean(distribution_1)
mean_2 = long_mean(distribution_2)
mean_3 = long_mean(distribution_3)
        

## 6. Introducing the Data ##

import pandas as pd

houses = pd.read_table("AmesHousing_1.txt")

print(houses.dtypes)

houses.head()

one = True
two = False
three = True 

## 7. Mean House Prices ##

def mean(distribution):
    sum_distribution = 0
    for value in distribution:
        sum_distribution += value
        
    return sum_distribution / len(distribution)

function_mean = mean(houses['SalePrice'])
pandas_mean = houses['SalePrice'].mean()

means_are_equal = function_mean == pandas_mean

## 8. Estimating the Population Mean ##

pop_mean = houses['SalePrice'].mean()
sampling_errors = [] 
sample_size = []
n = 5

for i in range(101):
    sample_size.append(n)
    sample = houses['SalePrice'].sample(n, random_state = i)
    sampling_errors.append(pop_mean - sample.mean())
    n += 29
    
plt.scatter(sample_size, sampling_errors)
plt.axhline(0)
plt.axvline(2930)
plt.xlabel("Sample size")
plt.ylabel("Sampling error")
plt.show()

## 9. Estimates from Low-Sized Samples ##

means = [] 
for i in range(10000):
    sample = houses['SalePrice'].sample(100, random_state = i)
    means.append(sample.mean())
    
plt.hist(means)
plt.axvline(houses['SalePrice'].mean())
plt.xlabel("Sample mean")
plt.ylabel("Frequency")
plt.xlim((0,500000))

## 11. The Sample Mean as an Unbiased Estimator ##

population = [3, 7, 2]
samples = [[3, 7] , [3, 2], 
           [7, 3], [7, 2],
           [2, 3], [2, 7]]

sample_means = 0

for sample in samples:
    sample_means += sum(sample)/len(sample)
sample_mean = sample_means/len(samples)

unbiased = sum(population)/len(population) == sample_mean