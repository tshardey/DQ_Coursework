## 3. Statistical significance ##

import numpy as np
import matplotlib.pyplot as plt

mean_group_a = np.mean(weight_lost_a)
print(mean_group_a)
mean_group_b = np.mean(weight_lost_b)

plt.hist(weight_lost_a)
plt.show()
plt.hist(weight_lost_b)
plt.show()

## 4. Test statistic ##

mean_difference = mean_group_b - mean_group_a 
print(mean_difference)

## 5. Permutation test ##

import numpy as np
import matplotlib.pyplot as plt

mean_difference = 2.52
print(all_values)

mean_differences = []

for i in range(0, 1000):
    group_a = []
    group_b = []
    for value in all_values:
        test_value = np.random.rand()
        if test_value >= 0.5:
            group_a.append(value)
        else:
            group_b.append(value)
    iteration_mean_difference = np.mean(group_b) - np.mean(group_a)
    mean_differences.append(iteration_mean_difference)

    
plt.hist(mean_differences)
plt.show()

## 7. Dictionary representation of a distribution ##

sampling_distribution = {}


for difference in mean_differences:
    if sampling_distribution.get(difference, False):
        sampling_distribution[difference] = sampling_distribution.get(difference) + 1
    else:
        sampling_distribution[difference] = 1

## 8. P value ##

frequencies = []

for key, value in sampling_distribution.items():
    if key >= 2.52:
        frequencies.append(value)
p_value = sum(frequencies)/1000