import numpy as np
import matplotlib.pyplot as plt

def a_lin(theta, M, d, v, f0):
    a = (np.ones((M,1),dtype="complex_"))
    for i in range(0,M):
        a[i] = a[i] * np.exp(-1j*(i)*d/v*np.sin(theta)*2*np.pi*f0)
        
    return a


def a_lin_multiplesources(theta, M, d, v, f0):
    a = np.ones((M,len(theta)),dtype="complex_")
    for j in range(0,len(theta)):
        for i in range(0,M):
            a[i][j] = np.exp(-1j*(i)*d/v*np.sin(theta[j])*2*np.pi*f0)
        
    return a

theta = [np.pi/2]

print(a_lin(np.pi/2,3,1,340,60))
print((a_lin_multiplesources(theta,3,1,340,60)))

