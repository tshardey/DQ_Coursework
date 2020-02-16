## 1. Why Learn Calculus? ##

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 301)

y = -(x**2)+ 3*x -1

plt.plot(x,y)

## 4. Math Behind Slope ##

def slope(x1,x2,y1,y2):
    ## Finish the function here
    slope = (y1-y2)/(x1-x2)
    return slope

slope_one = slope(0,4,1,13)
slope_two = slope(5,-1,16,-2)

## 6. Secant Lines ##

import seaborn
seaborn.set(style='darkgrid')

def draw_secant(x_values):
    x = np.linspace(-20,30,100)
    y = -1*(x**2) + x*3 - 1
    plt.plot(x,y)
    # Add your code here.
    y_value =[]
    for value in x_values:
        y_value.append(-1*(value**2) + value*3 - 1)
    plt.plot(x_values, y_value)
    plt.show()
    
draw_secant([3,5])
draw_secant([3,10])
draw_secant([3,15])