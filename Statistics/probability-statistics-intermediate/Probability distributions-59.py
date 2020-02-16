## 3. Bikesharing distribution ##

import pandas
bikes = pandas.read_csv("bike_rental_day.csv")

bikes_total = bikes.shape[0]
bikes_over_5000 = bikes[bikes['cnt']>5000].shape[0]
prob_over_5000 = bikes_over_5000/bikes_total

## 4. Computing the distribution ##

from math import factorial

# Each item in this list represents one k, starting from 0 and going up to and including 30.
outcome_counts = list(range(31))

def potential_outcomes(p, N, k):
    num1 = (p**k) * ((1-p)**(N-k))
    num2 = factorial(N)/(factorial(k)*factorial(N-k))
    return num1*num2

outcome_probs = []
for count in outcome_counts:
    outcome_probs.append(potential_outcomes(0.39, 30, count))

## 5. Plotting the distribution ##

import matplotlib.pyplot as plt

# The most likely number of days is between 10 and 15.
plt.bar(outcome_counts, outcome_probs)
plt.show()

## 6. Simplifying the computation ##

import scipy
from scipy import linspace
from scipy.stats import binom
import matplotlib.pyplot as plt

# Create a range of numbers from 0 to 30, with 31 elements (each number has one entry).
outcome_counts = linspace(0,30,31)

outcome_probs = binom.pmf(outcome_counts, 30, 0.39)

plt.bar(outcome_counts, outcome_probs)
plt.show()

## 8. Computing the mean of a probability distribution ##

dist_mean = 30 * 0.39

## 9. Computing the standard deviation ##

from math import sqrt
dist_stdev = sqrt(30*0.39*(1-0.39))

## 10. A different plot ##

# Enter your answer here.
import scipy
from scipy import linspace
from scipy.stats import binom
import matplotlib.pyplot as plt

binom_10 = binom.pmf(list(linspace(0, 10, 11)), 10, 0.39)
binom_100 = binom.pmf(list(linspace(0, 100, 101)), 100, 0.39)



plt.bar(list(linspace(0, 10, 11)), binom_10)
plt.show()
plt.bar(list(linspace(0, 100, 101)), binom_100)
plt.show()

## 11. The normal distribution ##

# Create a range of numbers from 0 to 100, with 101 elements (each number has one entry).
outcome_counts = scipy.linspace(0,100,101)

# Create a probability mass function along the outcome_counts.
outcome_probs = binom.pmf(outcome_counts,100,0.39)

# Plot a line, not a bar chart.
plt.plot(outcome_counts, outcome_probs)
plt.show()

## 12. Cumulative density function ##

import matplotlib.pyplot as plt
outcome_counts = linspace(0,30,31)

dist = binom.cdf(outcome_counts, 30, 0.39)

plt.plot(dist)

## 14. Faster way to calculate likelihood ##

left_16 = binom.cdf(16, 30, 0.39)
right_16 = 1- left_16