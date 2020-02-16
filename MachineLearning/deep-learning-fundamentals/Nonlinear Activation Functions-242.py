## 2. ReLU Activation Function ##

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
x = np.linspace(-2, 2, 20)

def relu(value):
    relu_y = []
    for x in value:
        if x<=0: 
            relu_y.append(0)
        else:
            relu_y.append(x)
    return relu_y

relu_y = relu(x)
print(x, relu_y)

plt.plot(x, relu_y)
    

## 3. Trigonometric Functions ##

x = np.linspace(-2*np.pi, 2*np.pi, 100)

tan_y = np.tan(x)

print(x, tan_y)
plt.plot(x, tan_y)

## 5. Hyperbolic Tangent Function ##

x = np.linspace(-40, 40, 100)

tanh_y = np.tanh(x)

print(x, tanh_y)
plt.plot(x, tanh_y)