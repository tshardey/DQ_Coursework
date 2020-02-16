## 1. Introduction ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

features = ['Wood Deck SF', 'Fireplaces', 'Full Bath', '1st Flr SF', 'Garage Area',
       'Gr Liv Area', 'Overall Qual']

X = train[features]
y = train['SalePrice']

term_1 = np.linalg.inv(np.dot(np.transpose(X), X))
term_2  = np.dot(np.transpose(X), y)

a = np.dot(term_1, term_2)
