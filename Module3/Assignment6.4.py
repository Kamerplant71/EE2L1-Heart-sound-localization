import numpy as np
import matplotlib.pyplot as plt

from Functions import a_lin, a_lin_multiplesources


def matchedbeamformer(Rx, th_range, M, d, v, f0):
    #Create empty Py
    Py = np.zeros((len(th_range)),dtype="complex_")

    #Loop for each theta
    for i in range(0,len(th_range)):
        #Calculate a_theta
        a_theta= a_lin(th_range[i], M, d, v, f0)
        a_H_theta = np.reshape( a_theta.conj(), (1,M) )
        #Calculate Power
        print(np.shape(a_H_theta))
        print(np.shape(a_theta))
        py= abs(np.matmul(np.matmul(a_H_theta,Rx),a_theta))
        Py[i] = py[0][0]

    return Py

#Initialize Constants
theta0= [0,np.pi/12]
Q = len(theta0)
SNR = 10
th_range = np.linspace(-np.pi/2,np.pi/2, 100)
M=7
d= 1
v= 340
f0= 140

#Calculate Rx
sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_lin_multiplesources(theta0, M, d, v, f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix
print(np.shape(Rx))   
    
P_mbf = matchedbeamformer(Rx, th_range, M, d, v, f0)

print(np.shape(P_mbf))
plt.plot(th_range*180/np.pi,(P_mbf))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.show()