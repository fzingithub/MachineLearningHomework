# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:54:08 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:39:07 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import numpy as np  
import matplotlib.pyplot as plt  

def trainPerceptron(dataMat, labelMat, eta):
    m, n = np.shape(np.mat(dataMat))
    weight = np.zeros(n)
    bias = 0
    flag = True
    while flag:
        for i in range(m):
            if np.any(labelMat[i] * (np.dot(weight, dataMat[i]) + bias) <= 0):
                weight = weight + eta * np.dot(labelMat[i],dataMat[i])
                bias = bias + eta * labelMat[i]
                print("weight, bias: ", end="")
                print(weight, end="  ")
                print(bias)
                flag = True
                break
            else:
                flag = False

    return weight, bias

def generateData(N):
    x1 = np.random.uniform(-3,-1,N)
    y1 = np.random.uniform(-1,1,N)

    x2 = np.random.uniform(1,3,N)
    y2 = np.random.uniform(-1,1,N)


    x=[]
    y=[]
    x_0=[]
    y_0=[]
    x_1=[]
    y_1=[]
    z=[]
    for i in range(N):
        if (x1[i]+2)**2+y1[i]**2 <1:
            x.append(x1[i])
            y.append(y1[i])
            x_0.append(x1[i])
            y_0.append(y1[i])
            z.append(1)
        if (x2[i]-2)**2+y2[i]**2 <1:
            x.append(x2[i])
            y.append(y2[i])
            x_1.append(x2[i])
            y_1.append(y2[i])
            z.append(-1)
    

    trainData = np.zeros((len(x),2))
    for i in range(len(x)):
        trainData[i][0]=x[i]
        trainData[i][1]=y[i]

    label = np.array(z)
    return trainData,label,x_0,y_0,x_1,y_1

def drawing(x_0,y_0,x_1,y_1):
    plt.subplots_adjust(right=0.825)
    ax=plt.gca()

    xx = (np.arange(-1, 1, 0.1)).T
    yy = (-weight[0]/weight[1])*xx-bias/weight[1]

    x=[]
    y=[]
    for i in range(len(yy)):
        if yy[i] > -3.0 and yy[i] < 3.0:
            x.append(xx[i])
            y.append(yy[i])
  
    ax.plot(x, y,'-',alpha = 1,label = 'Separating line')
    plt.scatter(x_0, y_0, alpha=0.6,label = 'Positive',color='green')
    plt.scatter(x_1, y_1, alpha=0.6,label = 'Negative',color='red')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_xticks(np.linspace(-4,4,9))
    ax.set_yticks(np.linspace(-3,3,7))

    plt.legend(loc='upper right')
    plt.savefig('Separated.eps',dpi=2000)
    plt.show()    
    
    
if __name__=='__main__':
    trainData,label,x_0,y_0,x_1,y_1 = generateData(180)
    weight, bias = trainPerceptron(trainData, label, 0.05)
    
    drawing(x_0,y_0,x_1,y_1)