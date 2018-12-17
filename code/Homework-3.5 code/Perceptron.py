# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:22:50 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import  numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')     #delete ' ',\n  split with '\t'
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def trainPerceptron(dataMat, labelMat, eta):
    m, n = np.shape(np.mat(dataMat))
    weight = np.zeros(n)
    bias = 0
    flag = True
    t = 0
    while flag:
        for i in range(m):
            if np.any(labelMat[i] * (np.dot(weight, dataMat[i]) + bias) <= 0):
                weight = weight + eta * np.dot(labelMat[i],dataMat[i])
                bias = bias + eta * labelMat[i]
                print("weight, bias: ", end="\t")
                print(weight, end="\t")
                print(bias,end="\t")
                t =t+1
                print("#iteration: ",t,)
                flag = True
                break
            else:
                flag = False

    return weight, bias


def drawing(dataArr,labelArr,w,b):
    n = np.shape(labelArr)[0] 
    xcord1 = []; ycord1 = []   
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelArr[i])== 1:
            xcord1.append(dataArr[i][0])
            ycord1.append(dataArr[i][1])
        else:
            xcord2.append(dataArr[i][0])
            ycord2.append(dataArr[i][1]) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=40, c='yellow', marker='s',label='class 1')
    ax.scatter(xcord2, ycord2, s=40, c='green',label='class -1')	

    ax.legend(loc='best')

    x = np.arange(-5, 9, 0.1)
    y = (-weight[0]/weight[1])*x-b/weight[1]
    
    ax.plot(x, y,'-')
    plt.xlabel('X1'); plt.ylabel('X2')    
    plt.title("data $\omega_2$ $\omega_3$")
    plt.savefig('w2_w3.eps',dpi=2000)

    plt.show()
    
def drawing1(dataArr,labelArr,w,b):
    n = np.shape(labelArr)[0] 
    xcord1 = []; ycord1 = []   
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelArr[i])== 1:
            xcord1.append(dataArr[i][0])
            ycord1.append(dataArr[i][1])
        else:
            xcord2.append(dataArr[i][0])
            ycord2.append(dataArr[i][1]) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=40, c='yellow', marker='s',label='class 1')
    ax.scatter(xcord2, ycord2, s=40, c='green',label='class -1')	

    ax.legend(loc='best')

    x = np.arange(-4, 9, 0.1)
    y = (-weight[0]/weight[1])*x-b/weight[1]
    
    ax.plot(x, y,'-')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.title("data $\omega_1$ $\omega_2$")
    plt.savefig('w1_w2.eps',dpi=2000)
    plt.show()  

if __name__ == "__main__":
    
#    dataMat, labelMat = generate_dataset_static()
    dataMat, labelMat = loadDataSet('w1_w2.txt')
    weight, bias = trainPerceptron(dataMat, labelMat, 1)
    drawing1(dataMat, labelMat, weight, bias)

    dataMat, labelMat = loadDataSet('w2_w3.txt')
    weight, bias = trainPerceptron(dataMat, labelMat, 1)
    drawing(dataMat, labelMat, weight, bias)

