# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 08:49:14 2018

@author: zhe

E-mail: 1194585271@qq.com
"""
#1.7 Plot the curve of the function sin(wx) for x ranged from -10 to +10 
#for a fixed w=1; repeat doing it for w=1:10. Please notice the change of the 
#curve with respect to the angle frequency w.
import numpy as np
import matplotlib.pyplot as plt

for w in range(1,11,1):
    x=np.arange(-10,10,0.01)
    y=np.sin(w*x)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,y,'g',linewidth=1,label=('w = ',w))
    plt.legend(loc='center')
    plt.show()
  




  
#1.8 Calculate the FFT of the function sin(wx) above with fft function in
# matlab and see the change of the spectrum with respect to w.
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


x=np.linspace(0,2*np.pi,500)      

#y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)

y = np.sin(4*x)

yy=fft(y)                     #快速傅里叶变换
yreal = yy.real               # 获取实数部分
yimag = yy.imag               # 获取虚数部分

yf=abs(fft(y))                # 取绝对值
yf1=abs(fft(y))/len(x)          #归一化处理
yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

xf = np.arange(len(y))        # 频率
xf1 = xf
xf2 = xf[range(int(len(x)/2))]  #取一半区间


plt.subplot(221)
plt.plot(x,y,label='w=4')   
plt.title('Original wave')
plt.legend(loc='center')

plt.subplot(222)
plt.plot(xf,yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

plt.subplot(223)
plt.plot(xf1,yf1,'g')
plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

plt.subplot(224)
plt.plot(xf2,yf2,'b')
plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')

plt.show()




#1.9 Two random variables x and y are statistically independent, and each 
#follows [0,1] uniform distribution respectively. Please draw the scatter 
#plot of the random vector (x,y), and the scatter plot of random vector
# (0.1x+0.9y, 0.9x+0.1y), to see the scatter plot change. What it will 
# be when x and y are Gaussian distributed N(0,1), and what it will be
# when (0.1,0.9) is changed to (0.4,0.6).
# 需导入要用到的库文件
import numpy as np  # 数组相关的库
import matplotlib.pyplot as plt  # 绘图库

N = 1000
x = np.random.rand(N)  # 包含1000个均匀分布的随机值的横坐标数组，大小[0, 1]
y = np.random.rand(N)  # 包含1000个均匀分布的随机值的纵坐标数组

plt.scatter(x, y, alpha=0.6,label = 'x,y-U(0,1)')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.show()


x1 = 0.1*x+0.9*y
y1 = 0.9*x+0.9*y
plt.scatter(x1, y1, alpha=0.6,label='$x_1 = 0.1x+0.9y$\n$y_1 = 0.9x+0.9y$')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.xlabel('$x_1$')
plt.ylabel('$y_1$')
plt.legend(loc='best')
plt.show()

x2 = 0.4*x+0.6*y
y2 = 0.6*x+0.4*y
plt.scatter(x2, y2, alpha=0.6,label='$x_2 = 0.4x+0.6y$\n$y_2 = 0.6x+0.4y$')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.xlabel('$x_2$')
plt.ylabel('$y_2$')
plt.legend(loc='best')
plt.show()



mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance

x, y = np.random.multivariate_normal(mean, cov,2000).T

plt.scatter(x, y, alpha=0.6,label='$(x,y)-G(\mu,\Sigma)$')  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()









#1.10 Generate two independent images, an image whose pixel intensities are
# uniformly distributed in [0,1] and an image whose pixel intensities are
# Gaussian distributed with means of mu=0 and variance of delt=0.01. (a) 
# Show the images with matlab function of imagesc respectively; (b) Calculate 
# the addition of the two images and show the images with matlab function of
# imagesc; (c) Calculate the multiplication of the two images and show the 
# images with matlab function of the imagesc; (d) Do the above three for
# delt=0.01:0.02:1 for visualizing the changes of the addition and
# multiplication of the two images respectively.

import numpy as np
import matplotlib.pyplot as plt
#图像
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 392

center_x = IMAGE_WIDTH/2
center_y = IMAGE_HEIGHT/2

R = np.sqrt(center_x**2 + center_y**2)

Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

# 利用 for 循环 实现
for i in range(IMAGE_HEIGHT):
    for j in range(IMAGE_WIDTH):
        dis = np.sqrt((i-center_y)**2+(j-center_x)**2)
        Gauss_map[i, j] = np.exp(-0.5*dis/R)

# 直接利用矩阵运算实现

mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

x1 = np.arange(IMAGE_WIDTH)
x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

y1 = np.arange(IMAGE_HEIGHT)
y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
y_map = np.transpose(y_map)

Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)

Gauss_map = np.exp(-0.5*Gauss_map/R)

# 显示和保存生成的图像
plt.figure()
plt.imshow(Gauss_map, plt.cm.gray)
plt.imsave('imageGaussian.jpg', Gauss_map, cmap=plt.cm.gray)
plt.show()



#==============================================================================
twoD = np.zeros((392,512))

for i in range(IMAGE_HEIGHT):
    for j in range(IMAGE_WIDTH):
        twoD[i][j] = np.random.ranf(1) 

plt.figure()
plt.imshow(twoD, plt.cm.gray)
plt.imsave('imageUniform.jpg',twoD, cmap=plt.cm.gray)
plt.show()

#==============================================================================
plus = twoD + Gauss_map

plt.figure()
plt.imshow(plus, plt.cm.gray)
plt.imsave('imagePlus.jpg',plus, cmap=plt.cm.gray)
plt.show()

#==============================================================================
multiply = twoD * Gauss_map

plt.figure()
plt.imshow(multiply, plt.cm.gray)
plt.imsave('imageMultiply.jpg',multiply, cmap=plt.cm.gray)
plt.show()




#==============================================================================




#三维高斯可视化
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

x, y = np.mgrid[-2:2:200j, -2:2:200j]
z=(1/2*np.pi*3**2)*np.exp(-(x**2+y**2)/2*3**2)
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)#绘面

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()



