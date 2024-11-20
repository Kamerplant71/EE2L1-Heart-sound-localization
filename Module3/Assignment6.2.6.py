import numpy as np
import matplotlib.pyplot as plt

from Assignment6_2_3 import a_lin


theta = np.linspace(-np.pi/2,np.pi/2,100)
M = 7
Delta = 1/2
f0 = 500
v=340
d = v*Delta/f0

a_theta0 = np.reshape(a_lin(0,M,d,v,f0),(7,1))


Py = np.zeros(len(theta))




for i in range(0,len(theta)-1):
    rowi = np.reshape( a_lin(theta[i],M,d,v,f0).conj(), (1,M) )
    py = abs(np.matmul(rowi,a_theta0))**2
    Py[i] = py[0][0]

print(np.shape(Py))

plt.plot(theta*180/np.pi,abs(Py))

plt.show()