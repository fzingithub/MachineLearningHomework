# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:33:06 2018

@author: zhe

E-mail: 1194585271@qq.com
"""
import numpy as np


def metrics(a, b): 
    return np.linalg.norm(a - b,2)   #默认是二范数
 

def gaussian (x, mu, sigma):
    return np.exp(- metrics(mu, x)**2/ (2 * sigma**2))



InData = np.array([[0,0],[0,1],[1,0],[1,1]])
Label = np.array([[0],[1],[1],[0]])

phi = []
for i in range(4):
    for j in range(4):
        phi.append(gaussian(InData[j],InData[i],np.sqrt(1/2)))
        
phiArr = np.array(phi).reshape((4,4)).T

phiArrInv = np.linalg.inv(phiArr)

W = np.mat(phiArrInv)*np.mat(Label) 

print (phiArr)
        
print (phiArrInv)

print (W)
        

 
