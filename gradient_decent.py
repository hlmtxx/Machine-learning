
# coding: utf-8

# In[21]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x=np.loadtxt('ax.dat')
y=np.loadtxt('ay.dat')

m=x.shape[0]
x.shape=(m,1)
x=np.hstack([np.ones((m,1)),x])

loop_max=10000
epsilon=1e-6
theta=np.zeros(2)
alpha=0.07
diff=0
error=np.zeros(2)
count=0
finish=0

while count<loop_max:
    count+=1
    sum_m=np.zeros(2)
    for i in range(m):
        diff=(np.dot(theta,x[i])-y[i])*x[i]
        sum_m=sum_m+diff
    
    theta=theta-alpha*sum_m/m
    
    if np.linalg.norm(theta-error)<epsilon:
        finish=1
        break
    else:
        error=theta
print('loop count = %d' % count, '\tweight:',theta)

plt.plot(x[:,1],y,'g*')
plt.plot(x[:,1],np.dot(x,theta),'r')
plt.xlabel('Age in years')
plt.ylabel('Height in meters')
plt.legend(['Traing Data','Linear Regression'])
plt.show()
plt.close()
predict1 = np.dot([1, 3.5] ,theta)
predict2 = np.dot([1, 7] , theta)
print('Age=3.5, Height= ',predict1 )
print('Age=7, Height= ',predict2 )
t0=np.linspace(-3,3,100)
t1=np.linspace(-1,1,100)

t0.shape=(len(t0),1)
t1.shape=(len(t1),1)

T0,T1=np.meshgrid(t0,t1)
dif=0

J_vals=np.zeros((len(t0),len(t1)))
for i in range(len(t0)):
    for j in range(len(t1)):
        t=np.hstack([t0[i],t1[j]])
        dif=0
        for k in range(m):
            #dif=dif+(np.dot(x[k],t)-y[k])*(np.dot(x[k],t)-y[k])
            dif=dif+(np.dot(x[k],t)-y[k])**2
            
        J_vals[i,j]=0.5*dif/m

J_vals=J_vals.T

fig=plt.figure()
ax=fig.gca(projection='3d')
#ax=Axes3D(fig)
ax.plot_surface(T0,T1,J_vals)
plt.show()
plt.close()
plt.contour(T0,T1,J_vals,np.logspace(-2,2,15))
plt.xlabel('Theta0')
plt.ylabel('Theta1')
plt.show()

   
        

