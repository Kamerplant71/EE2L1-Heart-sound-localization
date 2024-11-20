import numpy as np
import matplotlib.pyplot as plt

def a_lin(theta, M, d, v, f0):
    a = np.transpose(np.ones((M),dtype="complex_"))

    for i in range(1,M+1):
        a[i-1] = a[i-1] * np.exp(-1j*(i-1)*d/v*np.sin(theta)*2*np.pi*f0)
        
    return a

print(a_lin(np.pi/2,3,1,340,60))

