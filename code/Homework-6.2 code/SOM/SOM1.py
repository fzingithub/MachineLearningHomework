# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:17:33 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.8, 0.6],[0.1736, -0.9848], [0.707, 0.707], [0.342, -0.9397], [0.6, 0.8]])

X_radian = np.arctan2(X[:, 1], X[:, 0])
m = X.shape[0]
r = np.ones(m)

ax = plt.subplot(111, projection='polar')
ax.plot(X_radian, r, 'bo')
#plt.show()

iters_num = 20
learning_rate = 0.5
W = np.array([0, -np.pi])

W_r = np.ones(W.shape[0])
ax.plot(W, W_r, 'r*')
#plt.show()


for i in range(iters_num):
    for x in X_radian:
        d = np.abs(x - W)
#        print(x - W[1])
        if d[0] < d[1]:
            W[0] += learning_rate * (x - W[0])
        else:
            W[1] += learning_rate * (x - W[1])
#        print('iters{%d}' % i)
#        print("W[0] = ", W[0]*180/np.pi)
#        print("W[1] = ", W[1]*180/np.pi)
        
W_r = np.ones(W.shape[0])
ax.plot(W, W_r, 'r^')
plt.show()