import numpy as np
import matplotlib.pyplot as plt
from Functions import a_lin, a_lin_multiplesources



def a_lin(theta, M, d, v, f0):
    #Create empty complex vector
    a = (np.ones((M,1),dtype="complex_"))
    for i in range(0,M):
        #Compute each entry for vector
        a[i] = a[i] * np.exp(-1j*(i)*d/v*np.sin(theta)*2*np.pi*f0)
        
    return a


def a_lin_multiplesources(theta, M, d, v, f0):
    #Create matrix with each vector representing one theta
    a = np.ones((M,len(theta)),dtype="complex_")
    for j in range(0,len(theta)):
        #Same for a_lin for each theta[j]
        for i in range(0,M):
            a[i][j] = np.exp(-1j*(i)*d/v*np.sin(theta[j])*2*np.pi*f0)
        
    return a


theta = np.pi/2
M=3
d=1
v = 340
f0 =60

print(a_lin(theta,M,d,v,f0))

theta = [0,np.pi/2]
print((a_lin_multiplesources(theta,M,d,v,f0)))

