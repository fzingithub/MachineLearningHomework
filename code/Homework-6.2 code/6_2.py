# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:39:50 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

from minisom import MiniSom
import matplotlib.pyplot as plt
import numpy as np
   
def generateData(N):
    x3 = np.random.uniform(-1,1,N)
    y3 = np.random.uniform(0,1,N)
    x=[]
    y=[]
    for i in range(N):
        if x3[i]-y3[i]+1 > 0 and -x3[i]-y3[i]+1 > 0 and y3[i]>0:
            x.append(x3[i])
            y.append(y3[i])
            
    trainData = np.zeros((len(x),2))
    for i in range(len(x)):
        trainData[i][0]=x[i]
        trainData[i][1]=y[i]

    return trainData

# 数据可视化
def scat_data(weights,size=(8,4)):
    plt.figure(figsize=size)
    plt.scatter(weights[:,0], weights[:,1],s=6,alpha = 0.6)
    plt.savefig('data.eps',dpi = 1000)
    
    
data = generateData(3000)
#scat_data(data)

data_radian = np.arctan2(data[:, 1], data[:, 0])
m = data.shape[0]
r = np.ones(m)

ax = plt.subplot(111, projection='polar')
ax.plot(data_radian, r, 'go')
#plt.savefig('polar.eps',dpi = 1000)

H = 65
W = 1
som = MiniSom(H, W, input_len=2, sigma=0.3, learning_rate=0.5) # initialization of 65x1 SOM
#som.random_weights_init(data)
initweight = som._weights
som.train_random(data, 400) # trains the SOM with 100 iterations

weights = som.get_weights()
mappings = som.win_map(data)

weights = weights.reshape(H,2)

weights_radian = np.arctan2(weights[:, 1], weights[:, 0])
W_r = np.ones(weights_radian.shape[0])
ax.plot(weights_radian, W_r, 'r^')
#plt.savefig('40000_.eps',dpi=1000)

