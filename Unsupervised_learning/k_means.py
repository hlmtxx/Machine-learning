
# coding: utf-8

# In[2]:


import numpy as np
from scipy import misc
from scipy import sparse as sps
import matplotlib.pyplot as plt
from random import sample
import math



A=misc.imread('b_small.tiff',mode='RGB')
B=misc.imread('b_small.tiff',mode='RGB')
lenth=B.shape[0]
width=B.shape[1]

x_size=A.shape[0]
y_size=A.shape[1]
k=8

c=np.zeros((k,3))
class_data=np.zeros((x_size,y_size))
loop=0
epsilon=1e-3
error=np.zeros((k,3))

c[0][0]=107.84084281
c[0][1]=107.23750851  
c[0][2]=61.6090697
for i in range(1,k):
    distance_record=np.zeros((x_size,y_size))
    for j in range(x_size):
        for l in range(y_size):
            distance=np.zeros(i)
            for m in range(i):
                
                distance[m]=np.linalg.norm(c[m]-A[j][l])
            
            distance_record[j][l]=np.min(distance)

    
    index=np.where(distance_record==np.max(distance_record))
    
    c[i]=A[index[0][0]][index[1][0]]
    
while loop<=100:    
    loop+=1

    for i in range(x_size):
        for j in range(y_size):
            distance=np.zeros(k)
            for l in range(k):
                distance[l]=np.linalg.norm(c[l]-A[i][j])
            Min=np.argmin(distance)
            class_data[i][j]=Min
            #print class_data[i][j]

    c=np.zeros((k,3))
    
    for l in range(k):
        count=0
        for i in range(x_size):
            for j in range(y_size):
                if class_data[i][j]==l:
                    count+=1
                    c[l]=c[l]+A[i][j]
        if count==0:
            x[l]=sample(range(0,x_size), 1)
            y[l]=sample(range(0,y_size),1)
            c[l]=A[x[l]][y[l]]
        else:
            c[l]=c[l]/count
    if np.linalg.norm(c-error)<epsilon:
        break
    else:
        error=c  
print loop

for i in range(lenth):
    for j in range(width):
        distance=np.zeros(k)
        for l in range(k):
            distance[l]=np.linalg.norm(c[l]-B[i][j])
        Min=np.argmin(distance)
        B[i][j]=c[Min]    
print c
plt.imshow(B)  
plt.savefig('kmeans.png')
  


          
            
            
            

