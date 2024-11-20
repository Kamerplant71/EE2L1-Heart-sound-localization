import numpy as np
import matplotlib.pyplot as plt

def a_lin(theta, M, d, v, f0):
    a = np.reshape(np.ones((M),dtype="complex_"),(M,1))

    for i in range(0,M):
        a[i] = a[i] * np.exp(-1j*(i)*d/v*np.sin(theta)*2*np.pi*f0)
        
    return a

print((a_lin(np.pi/2,3,1,340,60)))

