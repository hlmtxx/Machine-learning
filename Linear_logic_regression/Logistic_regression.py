
# coding: utf-8

# In[8]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.special import expit
import math

x=np.loadtxt('cx.dat')
y=np.loadtxt('cy.dat')

m=y.shape[0]
x.shape=(m,2)
y.shape=(m,1)
x=np.hstack([np.ones((m,1)),x])

pos=np.nonzero(y)[0]
neg=np.where(y==0)[0]


plt.scatter(x[pos,1],x[pos,2],marker='+')
plt.scatter(x[neg,1],x[neg,2],facecolors='none',marker='o',color='r')


loop_max=10000
epsilon=1e-8
theta=np.zeros((3,1))
error=np.zeros((3,1))
count=0
finish=0
J=[]

while count<=loop_max:
    count+=1
    h=expit(np.dot(x,theta))
    deltaJ=(1.0/m)*np.dot(x.T,(h-y))
    sum_m=0
    H=np.zeros((3,3))
    for i in range(m):
        
        sum_m=sum_m+y[i]*math.log(h[i])+(1-y[i])*math.log(1-h[i])
        H=H+(1.0/m)*h[i]*(1-h[i])*np.dot(x[i].reshape(3,1),x[i].reshape(1,3))
    sum_m=(-1.0/m)*sum_m
    J.append(sum_m)
    theta=theta-np.dot(np.linalg.inv(H),deltaJ)
    if np.linalg.norm(theta-error)<epsilon:
        finish=1
        break
    else:
        error=theta
        

a=x[:,1]
b=(-theta[0]-theta[1]*a)/theta[2]

plt.plot(a,b,'r')

plt.show()
print J
plt.plot(range(count),J)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J') 
plt.show()
    
print 'loop count = %d' % count ,  ', weight:',theta.T

predict=1-expit(np.dot([1,20,80],theta))
        
print('the probability that a student with a score of 20 on Exam 1 and a score of 80 on Exam 2 will not be admitted is %s' % predict[0] )

