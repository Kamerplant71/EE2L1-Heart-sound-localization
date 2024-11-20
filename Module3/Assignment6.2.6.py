import numpy as np
import matplotlib.pyplot as plt

from Module3.Assignment6_2_3 import a_lin


theta = np.linspace(-np.pi/2,np.pi/2,100)
M = 7
Delta = 1/2
f0 = 500
v=340
d = v*Delta/f0

a_theta0 = a_lin(0,M,d,v,f0)

Py = np.zeros(len(theta))

Ah_theta= np.zeros(len(theta))
for i in range(0,len(theta)):

    y[i] = a_lin(theta[i], M, d, v, f0)
    
Py =  np.multiply(y.conj().T,a_theta0)

print(Py)