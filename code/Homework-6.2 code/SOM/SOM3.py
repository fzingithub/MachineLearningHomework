# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:48:25 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

# 由二维分布驱动一个二维的网络
# 输出层网络的大小有 H*W，个神经元组成
# 训练数据为二维向量，分量x1,x2均匀分布在｛－１，＋１｝之间


import numpy as np
import matplotlib.pyplot as plt


# 网络的辅助函数
# 欧式距离，返回一个数组
def distEclud(x, weights):
    dist = []
    for w in weights:
        d = np.linalg.norm(x-w)
        dist.append(d)
    return np.array(dist)
        
# 输出层网格的初始化（每个节点制定网格中的坐标）
# h,w　分别是输出层网格的高和宽
def init_grid(h,w):
    k = 0
    grid = np.zeros((h*w, 2))
    for i in range(h):
        for j in range(w):
            grid[k,:] = [i,j]
            k = k+1
    return grid

# 网络权值的初始化
# 权值向量的维度和向量的维度是相等的
def init_weights(h, w):
    # h,网格的高
    # w,网格的宽
    weights = np.random.uniform(-0.1,0.1,(h*w, 2))
    return weights

# 拓扑邻域半径衰减函数
def radius(n, t, r_0):
    # n是迭代的次数
    # t是一个时间常数
    # r_0 是半径的初始值，一般为网格的半径
    return r_0 * np.exp(-n / (t/np.log(r_0)))

# 学习效率的衰减函数
def learn_rate(n, t, r_0):
    # n 迭代的次数
    # t 一个常数
    # r_0 初始的学习效率
    return r_0 * np.exp(-n / (t/np.log(5.0)))


#可视化方法
def scat_weights(weights,size=(6,6)):
    plt.figure(figsize=size)
    plt.scatter(weights[:,0], weights[:,1],s=6)
    
def plot_weights(weights, h=10, w=10, size=(6,6)):
    x_axis = weights[:,0].reshape(h, w)
    y_axis = weights[:,1].reshape(h, w)
    plt.figure(figsize=size)    
    for i in range(h):
        plt.plot(x_axis[i], y_axis[i])
        plt.plot(x_axis.T[i], y_axis.T[i])
        
        


# 初始化
H=10; W=10   
grid = init_grid(int(H), W)       # 网格为10,10二维网格
weights = init_weights(int(H), W)

# 初始权值可视化

#scat_weights(weights)
plot_weights(weights, h=H, w=W, size=(6,6))


data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""

a = data.split(',')
x = np.mat([[float(a[i]), float(a[i + 1])] for i in range(1, len(a) - 1, 3)])

#plot_weights(np.array(x))



rads = []    # 记录半径的变化值
rates = []   # 记录学习效率的变化

for n in range(1,100001):
    r = radius(n, 4000, 6.0)
    rate = learn_rate(n, 4000, 0.1)
    rads.append(r)
    rates.append(rate)
    
    # 随机选择一个向量x
    k = np.random.randint(0,len(x))
    sample = x[k,:]
    
    # 根据最距离小化原则，找到获胜神经元
    # 计算每个权值向量与x的欧几里得距离
    dists = distEclud(sample, weights)  
    index = dists.argmin()
    
    # 计算获胜神经元在输出网格中的位置（坐标）
    # ceil() 返回大于输入值的整数
    d1 = np.ceil(index / H)
    d2 = np.mod(index , H)
    
    # 计算网格中其他神经元到激活神经元的侧向距离
    dist2 = distEclud(np.array([d1, d2]), grid)
    # 获取拓扑邻域有效半径内的神经元索引
    # dist2 &lt; r,得到一个布尔值数组
    # nonzero()返回数组中为真的索引
    index2 = (dist2 < r).nonzero()[0]
    
    # 更新拓扑邻域，有效半径内神经元的权值
    for j in index2:
        weights[j,:] = weights[j,:] + rate*(sample - weights[j,:])
    
    # 记录部分权值
    if n == 501:
        w_5 = weights.copy()
     
    if n % 10000 == 0:
        print ('%d0k is over'%(n/10000))

scat_weights(w_5)
plot_weights(w_5,h=H, w=W, size=(6,6))
        
scat_weights(weights)
plot_weights(weights,h=H, w=W, size=(6,6))


