import numpy as np
import matplotlib.pyplot as plt

from Assignment6_2_3 import a_lin, a_lin_multiplesources


def matchedbeamformer(Rx, th_range, M, d, v, f0):
    a_theta = a_lin_multiplesources(th_range, M ,d ,v, f0)
        
    a_H= a_theta.conj().T
    print(np.shape(a_theta))
    print(np.shape(a_H))
    Py1= np.matmul(a_H, Rx)
    Py2=np.matmul(Py1, a_theta)
    #Py= Py2[0][]+Py2[1][0]
    
    return Py2

theta0= [0,np.pi/12]
Q = len(theta0)
SNR = 10
th_range = np.linspace(-np.pi/2,np.pi/2, 100)
M=6
d= 1
v= 340
f0= 1000

sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_lin_multiplesources(theta0, M, d, v, f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix
print(np.shape(Rx))   
    
P_mbf = matchedbeamformer(Rx, th_range, M, d, v, f0)

print(np.shape(P_mbf))
print(P_mbf)
plt.plot(th_range, abs(P_mbf))
plt.show()
#P_mvdr = mvdr(Rx, th_range, M, d, v, f0)