## 2. Probability of renting bikes ##

import pandas
bikes = pandas.read_csv("bike_rental_day.csv")

# Find the number of days the bikes rented exceeded the threshold.
days_over_threshold = bikes[bikes["cnt"] > 2000].shape[0]
# Find the total number of days we have data for.
total_days = bikes.shape[0]

# Get the probability that more than 2000 bikes were rented for any given day.
probability_over_2000 = days_over_threshold / total_days
print(probability_over_2000)

days_over_4000 = bikes[bikes['cnt'] > 4000].shape[0]
probability_over_4000 = days_over_4000/total_days

## 4. Calculating probabilities ##

# Enter your code here.
coin_1_prob =  0.5 ** 3 * 3


## 6. Calculating the number of combinations ##

sunny_1_combinations = 5

## 8. Finding the number of combinations ##

import math
def find_outcome_combinations(N, k):
    # Calculate the numerator of our formula.
    numerator = math.factorial(N)
    # Calculate the denominator.
    denominator = math.factorial(k) * math.factorial(N - k)
    # Divide them to get the final value.
    return numerator / denominator

combinations_7 = find_outcome_combinations(10, 7)
combinations_8 = find_outcome_combinations(10, 8)
combinations_9 = find_outcome_combinations(10, 9)

## 10. Calculating the probability of one combination ##

prob_combination_3 = .7*.7*.7*.3*.3

## 12. Function to calculate the probability of a single combination ##

p = .6
q = .4

def single_combination(N, k):
    return (p ** k) * (q ** (N-k))

prob_8 = single_combination(10, 8) * find_outcome_combinations(10,8)
prob_9 = single_combination(10, 9)* find_outcome_combinations(10,9)
prob_10 = single_combination(10, 10)* find_outcome_combinations(10,10)
    
