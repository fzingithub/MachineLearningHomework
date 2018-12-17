# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:07:05 2018

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

data = generateData(200)
som = MiniSom(10, 10, input_len=2, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.random_weights_init(data)
som.train_random(data, 5000) # trains the SOM with 100 iterations

weights = som.get_weights()
mappings = som.win_map(data)

#Visualizing the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)  #平均距离
colorbar()
for i,x in enumerate(data):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,'^','r',markersize= 10)


print (som.get_weights)
print (som.win_map(data))
print (som.distance_map().T)