
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
cluster_index=[]

for k in range(2,14):
    
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
    
    
    #distance1=[]
    dis_max=np.zeros(k)
    for l in range (k):
        count=0
        distance1=[]
        for i in range(x_size):
            for j in range (y_size):
                if class_data[i][j]==l:
                    distance1.append(np.linalg.norm(c[l]-A[i][j]))
                    count+=1
        
        dis_max[l]=max(distance1)*count
    #dis_max[k-2]=max(distance1)
    #print dis_max[k-2]
    print (sum(dis_max))/(x_size*y_size)
    cluster_index.append((sum(dis_max))/(x_size*y_size))
plt.plot(range(2,14),cluster_index)
plt.xlabel('Number of clusters')
plt.ylabel('Cluster index') 
plt.show()


          
            
            
            


# In[10]:


import numpy as np
from scipy import misc
from scipy import sparse as sps
import matplotlib.pyplot as plt
from random import sample
import math

a=[165.76491163805687,
144.24686430385057,
117.96089909902774,
106.9215299284633,
99.91443465700576,
95.42445889515884,
81.55964450857859,
79.76114829910874,
78.87910267343803,
76.87910267343803,
75.63235719075519,
75.13235719075519,
74.63235719075519]

plt.plot(range(2,15),a)
plt.xlabel('Number of clusters')
plt.ylabel('Cluster index') 
plt.show()

