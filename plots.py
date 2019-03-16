
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sarsa import *
from montecarlo import *

value=np.zeros((2,11,22))
counter=np.zeros((2,11,22))

for i in xrange(1000000):
    value, counter=montecarlo(value,counter)

bestval=np.amax(value,axis=0)
bestval=bestval.T
fig=plt.figure()
ha = fig.add_subplot(111, projection='3d')
x = range(10)
y = range(21)
X, Y = np.meshgrid(x, y)
print X.shape,Y.shape,bestval.shape
ha.plot_wireframe(X+1, Y+1, bestval[1:,1:])

ha.set_xlabel("dealer starting card")
ha.set_ylabel("player current sum")
ha.set_zlabel("value of state")
plt.show()

    

