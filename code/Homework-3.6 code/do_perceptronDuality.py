# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:39:07 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import numpy as np  # 数组相关的库
import matplotlib.pyplot as plt  # 绘图库

def generateData(N):
    x3 = np.random.uniform(-2,2,N)
    y3 = np.random.uniform(-2,2,N)

    x=[]
    y=[]
    x_0=[]
    y_0=[]
    x_1=[]
    y_1=[]
    x_2=[]
    y_2=[]
    z=[]
    for i in range(N):
        if x3[i]**2+y3[i]**2 <1:
            x.append(x3[i])
            y.append(y3[i])
            x_0.append(x3[i])
            y_0.append(y3[i])
            z.append(1)
        if x3[i]>0 and x3[i]**2+y3[i]**2>1.5**2 and x3[i]**2+y3[i]**2<4:
            x.append(x3[i])
            y.append(y3[i])
            x_1.append(x3[i])
            y_1.append(y3[i])
            z.append(-1)
        if x3[i]>-0.5 and x3[i]**2+y3[i]**2>1.2**2 and x3[i]**2+y3[i]**2<1.3**2:
            x_2.append(x3[i])
            y_2.append(y3[i])
            
    trainData = np.zeros((len(x),2))
    for i in range(len(x)):
        trainData[i][0]=x[i]
        trainData[i][1]=y[i]

    label = np.array(z)
    return trainData,label,x_0,y_0,x_1,y_1,x_2,y_2

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
#            print (K[j],deltaRow*deltaRow.T,j)
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


def trainModel(dataMatIn,labelMatIn,eta,K):
    dataMat = np.mat(dataMatIn); labelMat = np.mat(labelMatIn).transpose()
    b = 0; m,n = np.shape(dataMat)
    alpha = np.mat(np.zeros((m,1)))
    flag = True
    while flag:
        for i in range(m):
            if  labelMatIn[i]*(float(np.multiply(alpha,labelMat).T*K[:,i]) + b)<= 0:
                alpha[i] = alpha[i] + eta 
                b = b + eta * labelMat[i]
#                w = np.multiply(labelMat,alpha).T*dataMat
                print (i,alpha[i])
                flag = True
                break
            else:
                flag = False
#    w = (np.multiply(labelMat,alpha).T*dataMat).T
    return alpha

    

if __name__=='__main__':
    trainData,label,x_0,y_0,x_1,y_1,x_2,y_2 = generateData(1200)
    m = np.shape(trainData)[0]
    K = np.mat(np.zeros((m,m)))

    #kernel matrix
    for i in range(m):
        K[:,i] = kernelTrans(np.mat(trainData), np.mat(trainData[i,:]), ('rbf',1))

    
    alpha = trainModel(trainData,label,1,K)
    
  
    
    plt.subplots_adjust(right=0.825)
    ax=plt.gca()
    plt.scatter(x_2, y_2, alpha=1,label = 'separated',)
    plt.scatter(x_0, y_0, alpha=0.6,label = 'Positive',color='green')
    plt.scatter(x_1, y_1, alpha=0.6,label = 'Negative',color='red')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_xticks(np.linspace(-4,4,9))
    ax.set_yticks(np.linspace(-3,3,7))
    plt.legend(loc='upper right')
    plt.savefig('Inseparated.eps',dpi=2000)
    plt.show()