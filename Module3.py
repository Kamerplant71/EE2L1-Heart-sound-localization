import numpy as np
import matplotlib.pyplot as plt

def a_lin(theta, M, d, v, f0):
    a = np.transpose(np.ones((M),dtype="complex_"))

    for i in range(1,M+1):
        a[i-1] = a[i-1] * np.exp(-1j*(i-1)*d/v*np.sin(theta)*2*np.pi*f0)
        
    return a

#print(a_lin(np.pi/2,3,1,340,60))

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