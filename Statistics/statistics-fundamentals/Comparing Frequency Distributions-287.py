## 2. Grouped Bar Plots ##

import seaborn as sns
sns.countplot(x = 'Exp_ordinal', hue = 'Pos', data = wnba,
              order = ['Rookie', 'Little experience', 'Experienced', 'Very experienced', 'Veteran'],
              hue_order = ['C', 'F', 'F/C', 'G', 'G/F']
             )    

## 3. Challenge: Do Older Players Play Less? ##

sns.countplot(x='age_mean_relative', hue='min_mean_relative', data=wnba)
result='rejection'

## 4. Comparing Histograms ##

import matplotlib.pyplot as plt

plt.axvline(497, label='Average')
plt.legend()

## 5. Kernel Density Estimate Plots ##

wnba[wnba.Age >= 27]['MIN'].plot.kde(label = 'Old', legend = True)
wnba[wnba.Age < 27]['MIN'].plot.kde(label = 'Young', legend = True)
plt.axvline(497, label='Average')
plt.legend()

## 7. Strip Plots ##

sns.stripplot(x='Pos', y='Weight', data=wnba, jitter=True)

## 8. Box plots ##

sns.boxplot(x='Pos', y='Weight', data=wnba)

## 9. Outliers ##

iqr=29-22
lower_bound = 22-1.5*iqr
upper_bound = 29+1.5*iqr

outliers_high = len(wnba[wnba['Games Played'] > upper_bound])
outliers_low = len(wnba[wnba['Games Played'] < lower_bound])
sns.boxplot(wnba['Games Played'])