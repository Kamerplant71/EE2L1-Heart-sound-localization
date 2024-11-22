import numpy as np
import matplotlib.pyplot as plt

from Functions import a_lin

#Theta range from -90 to 90 degrees
theta = np.linspace(-np.pi/2,np.pi/2,1000)

M = 7
Delta = 1/2
f0 = 500
v=340
d = v*Delta/f0

#Create a_theta0 
a_theta0 = a_lin(np.pi/4,M,d,v,f0)

#Make Py an array with length number of calculated angles

Py = np.zeros(len(theta))

for i in range(0,len(theta)-1):
    #Calculate aH for each theta[i]
    aH = np.reshape( a_lin(theta[i],M,d,v,f0).conj(), (1,M) )

    #Calculate Power and add to list
    py = abs(np.matmul(aH,a_theta0))**2
    Py[i] = py[0][0]

print(np.shape(Py))

plt.plot(theta*180/np.pi,abs(Py))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial response beamformer")
plt.show()