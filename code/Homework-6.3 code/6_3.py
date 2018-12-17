# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:50:33 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

from minisom import MiniSom
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

data =np.array([[-7.82 , -4.58 , -3.97 ],
        [-6.68 , 3.16 , 2.71 ],
        [4.36 , -2.19 , 2.09 ],
        [6.72 , 0.88 , 2.80 ],
        [-8.64 , 3.06 , 3.50 ],
        [-6.87 , 0.57 , -5.45 ],
        [4.47 , -2.62 , 5.76 ],
        [6.73 , -2.01 , 4.18 ],
        [-7.71 , 2.34 , -6.33 ],
        [-6.91 , -0.49 , -5.68 ],
        [6.18 , 2.81 , 5.82 ],
        [6.72 , -0.93 , -4.04 ],
        [-6.25 , -0.26 , 0.56 ],
        [-6.94 , -1.22 , 1.13 ],
        [8.09 , 0.20 , 2.25 ],
        [6.81 , 0.17 , -4.15 ],
        [-5.19 , 4.24 , 4.04 ],
        [-6.38 , -1.74 , 1.43 ],
        [4.08 , 1.30 , 5.33 ],
        [6.27 , 0.93 , -2.78 ]])

H = 4
W = 2

lattice = []
for i in range(H):
    for j in range(W):
        lattice.append((i,j))

som = MiniSom(H, W, input_len=3, sigma=0.3, learning_rate=0.5) # initialization of 65x1 SOM
som.random_weights_init(data)

som.train_random(data, 100) # trains the SOM with 100 iterations

weights = som.get_weights()
#weights = weights.reshape((H*W,3))
mappings = som.win_map(data)

#print (len(mappings))



x, y, z = data[:,0], data[:,1], data[:,2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程

color = ['g','r','c','k','y','pink','salmon','teal']
for i in range(len(mappings)):
    for j in range(len(mappings[lattice[i]])):
        x, y, z = mappings[lattice[i]][j][0], mappings[lattice[i]][j][1], mappings[lattice[i]][j][2]
        ax.scatter(x, y, z, c=color[i%8])  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
#plt.savefig('4_2_1000.eps',dpi=1000)
plt.show()

