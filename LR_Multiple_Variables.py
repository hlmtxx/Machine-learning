
# coding: utf-8

# In[80]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x=np.loadtxt('bx.dat')
y=np.loadtxt('by.dat')

m=y.shape[0]
x.shape=(m,2)
y.shape=(m,1)
sigma=np.std(x,axis=0)
mu=np.mean(x,axis=0)
x=(x-mu)/sigma
x=np.hstack([np.ones((m,1)),x])


alpha=[0.01,0.03,0.09,0.1,0.3,0.9,1.0,1.2]
loop_max=50



record=[]

J=np.zeros((len(alpha),50))

for j in range(8):
   
    theta = np.zeros((3,1))
    
    for i in range(loop_max):
        J_1=(0.5/m)*np.dot((np.dot(x,theta)-y).T,(np.dot(x,theta)-y))
        J[j,i]=J_1[0,0]
        theta=theta-((alpha[j]/m)*np.dot((np.dot(x,theta)-y).T,x)).T
    
    
    record.append(theta)

plt.plot(range(loop_max),J[0],'r')
plt.plot(range(loop_max),J[1],'b')
plt.plot(range(loop_max),J[2],'g')
plt.plot(range(loop_max),J[3],'m')
plt.plot(range(loop_max),J[4],'c')
plt.plot(range(loop_max),J[5],'y')
plt.plot(range(loop_max),J[6],'k')
plt.plot(range(loop_max),J[7],'maroon')
    
plt.legend(['alpha=0.01','alpha=0.03','alpha=0.09','alpha=0.1','alpha=0.3','alpha=0.9','alpha=1.0','alpha=1.2'])
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')  
plt.show()

theta=record[5]
print('The learning rate is : %f The Weight Using Gradient Decent is : Theta0 = %s , Theta1 = %s , Theta2 = %s' % (alpha[5],theta[0],theta[1],theta[2]))

a=[1650,3]
b=(a-mu)/sigma
b=np.hstack([np.ones(1),b])



predict1=np.dot(b,theta)

print ('Prediction of my model for the price of a house with 1650 square feet and 3 bedrooms is : %s'% predict1)

theta=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
predict2=np.dot(b,theta)
print('The Weight Normal Equations is : Theta0 = %s , Theta1 = %s , Theta2 = %s' % (theta[0],theta[1],theta[2]))

print ('Prediction of my  normal equations theta  for the price of a house with 1650 square feet and 3 bedrooms is : %s'% predict2)



