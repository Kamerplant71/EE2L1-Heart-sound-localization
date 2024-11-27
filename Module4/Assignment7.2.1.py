import numpy as np
import matplotlib.pyplot as plt
from Functions import a_lin, a_lin_multiplesources


def music(Rx, Q, M, th_range, d, v, f0):
    U, S , Vh = np.linalg.svd(Rx)
    Un = U[:,Q:M-Q]
    Un_H = Un.conj().T

    Py= np.zeros(len(th_range),dtype='complex')
    for i in range(0,len(th_range)):
        A = a_lin(th_range[i],M,d,v,f0)
        Ah = A.conj().T
        py_denom = np.matmul(np.matmul( np.matmul(Ah,Un),Un_H ),A   )
        py = 1/ py_denom
        Py[i] = py[0][0]
    
    return Py


theta0= [0,np.pi/12]
Q = len(theta0)
SNR = 10
th_range = np.linspace(-np.pi/2,np.pi/2, 1000)
M=7
delta =0.5
v= 340
f0= 140
d = delta*v/f0

#Calculate Rx
sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_lin_multiplesources(theta0, M, d, v, f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix

Py_music = music(Rx,Q,M,th_range,d,v,f0)


plt.plot(th_range*180/np.pi,10*np.log10(abs(Py_music)))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial spectrum, MUSIC")

plt.show()
