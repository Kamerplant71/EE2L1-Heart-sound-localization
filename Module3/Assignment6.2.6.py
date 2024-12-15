import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from Functions import a_lin

#Theta range from -90 to 90 degrees
th_range = np.linspace(-np.pi/2,np.pi/2,500)

#Constants
M = 7
Delta = 1/2
f0 = 500
v=340
d = v*Delta/f0
Direction_S = 60 / 180 * np.pi


#Create a_theta0 
a_th_range0 = a_lin(Direction_S,M,d,v,f0)

#Make Py as an array with length equal to number of angles that are caclulated
Py = np.zeros(len(th_range))

for i in range(len(th_range)):
    #Calculate aH for each th_range[i]
    aH = np.reshape( a_lin(th_range[i],M,d,v,f0).conj(), (1,M) )

    #Calculate Power and add to list
    py = abs(np.matmul(aH,a_th_range0))**2
    Py[i] = py[0][0]

print(np.shape(Py))
plt.rcParams['figure.figsize'] = [7, 6]
plt.rcParams['figure.dpi'] = 150

plt.plot(th_range*180/np.pi,abs(Py))
plt.xlabel("Angle [deg]", fontsize =18)
plt.xticks(fontsize = 16)

plt.ylabel("Power ",fontsize= 19)
plt.yticks(fontsize = 16)

plt.title("Spatial response fixed beamformer with source at 25°", fontsize = 19)
plt.vlines(25,ymin=min(Py),ymax=max(Py),linestyles="dashed",colors="r")

plt.legend(['Power','Expected angle'],loc = 'upper left',fontsize=17)

plt.tight_layout()
plt.show()