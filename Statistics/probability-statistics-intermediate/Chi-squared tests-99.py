## 2. Calculating differences ##

female_diff = (10771-16280.5)/16280.5
male_diff = (21790-16280.5)/16280.5

## 3. Updating the formula ##

female_diff = (10771-16280.5)**2/16280.5
male_diff = (21790-16280.5)**2/16280.5

gender_chisq = female_diff + male_diff

## 4. Generating a distribution ##

import numpy as np
import matplotlib.pyplot as plt
chi_squared_values = []

for i in range(0,1000):
    random = np.random.random((32561,))
    gender_list = []
    for number in random:
        if number < 0.5:
            number = 0
        else:
            number = 1
        gender_list.append(number)
    female_count = sum(gender_list)
    male_count = 32561 - female_count
    male_diff = (male_count - 32561/2)**2/(32561/2)
    female_diff = (female_count - 32561/2)**2/(32561/2)
    chi_squared_values.append(male_diff+female_diff)

plt.hist(chi_squared_values)
plt.show()

## 6. Smaller samples ##

female_diff = (107.71 - 162.805)**2/162.805
male_diff = (217.90 - 162.805)**2/162.805

gender_chisq = female_diff + male_diff

## 7. Sampling distribution equality ##

import numpy as np
import matplotlib.pyplot as plt

chi_squared_values = []

for i in range(0,1000):
    random = np.random.random((300,))
    random_count = []
    for number in random:
        if number < 0.5:
            number = 0
        else:
            number = 1
        random_count.append(number)
    female_count = sum(random_count)
    male_count = 300 - female_count
    female_diff = (female_count-150)**2/150
    male_diff = (male_count-150)**2/150
    chi_squared_values.append(female_diff + male_diff)
plt.hist(chi_squared_values)
plt.show()
                             

## 9. Increasing degrees of freedom ##

observed = [27816, 3124, 1039, 311, 271, 32561]
expected = [26146.5, 3939.9, 944.3, 260.5, 1269.8, 32561]

race_chisq = 0

for i in range(0, len(observed)):
    race_chisq += (observed[i] - expected[i])**2/expected[i]

## 10. Using SciPy ##

from scipy.stats import chisquare
import numpy as np

observed = [27816, 3124, 1039, 311, 271, 32561]
expected = [26146.5, 3939.9, 944.3, 260.5, 1269.8, 32561]

race_pvalue = chisquare(observed, expected)