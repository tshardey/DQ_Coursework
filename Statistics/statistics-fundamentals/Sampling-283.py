## 3. Populations and Samples ##

question1 = 'population'
question2 = 'population'
question3 = 'sample'
question4 = 'population'
question5 = 'sample'

## 4. Sampling Error ##

import pandas as pd
wnba = pd.read_csv('wnba.csv')

print(wnba.head())
print(wnba.tail())
print(wnba.shape)

parameter = wnba['Games Played'].max()

sample = wnba['Games Played'].sample(random_state=1)
statistic = sample.max()

sampling_error = parameter-statistic

## 5. Simple Random Sampling ##

import pandas as pd
import matplotlib.pyplot as plt

wnba = pd.read_csv('wnba.csv')

pts_list = []

for i in range(0,100):
    sample = wnba['PTS'].sample(10, random_state=i)
    pts_list.append(sample.mean())
 
plt.scatter(list(range(1,101)), pts_list)
plt.axhline(wnba['PTS'].mean())
plt.show()

## 7. Stratified Sampling ##

wnba['Pts_per_game'] = wnba['PTS'] / wnba['Games Played']

position = wnba['Pos'].unique()
position_dic = {}

for job in position:
    subset = wnba[wnba['Pos']==job]
    pts_pos = subset['Pts_per_game'].sample(10, random_state=0).mean()
    position_dic[job] = pts_pos

position_most_points = max(position_dic, key=position_dic.get)

## 8. Proportional Stratified Sampling ##

wnba['Games Played'].value_counts(bins =3) 

strata = [[0,12], [13,22], [23,33]]

strata_dic = {}
for i in range(len(strata)):
    strata_dic[i]= wnba[wnba['Games Played'].between(strata[i][0], strata[i][1])]
    print(strata_dic[i].head())
pts_list = []
for k in range(100):
    first = strata_dic[0]['PTS'].sample(1, random_state=k)
    second = strata_dic[1]['PTS'].sample(2, random_state=k)
    third = strata_dic[2]['PTS'].sample(7, random_state=k)
    pts_list.append(pd.concat([first, second, third]).mean())
          
plt.scatter(range(1,101), pts_list)
plt.axhline(wnba['PTS'].mean())
plt.show()

## 9. Choosing the Right Strata ##

wnba['MIN'].value_counts(bins = 3, normalize = True)

## 10. Cluster Sampling ##

pick_four = pd.Series(wnba['Team'].unique()).sample(4, random_state=0)

pick_four_df = pd.DataFrame()

for team in pick_four:
    pick_four_df = pd.concat([pick_four_df, wnba[wnba['Team']==team]])
    
    
sampling_error_BMI = wnba['BMI'].mean() - pick_four_df['BMI'].mean() 
sampling_error_age = wnba['Age'].mean() - pick_four_df['Age'].mean() 
sampling_error_height = wnba['Height'].mean() - pick_four_df['Height'].mean() 
sampling_error_points = wnba['PTS'].mean() - pick_four_df['PTS'].mean() 