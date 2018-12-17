# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:25:53 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

from sklearn.svm import SVC
import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

for j in range(100):
    clf = SVC(C=1000000,kernel='poly',degree=j)
    clf.fit(X, y)
    trainingAccuracy = 0
    trainingAccuracy += sum(y == clf.predict(X))/len(X)
    print ('power = ',j,'\taccuracy = ',trainingAccuracy)