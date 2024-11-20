import numpy as np
import matplotlib.pyplot as plt

from Assignment6_2_3 import a_lin


def matchedbeamformer(Rx, th_range, M, d, v, f0):
    a_theta = np.zeros(M, len(th_range))
    for i in range(0, len(th_range)-1):
        columni= a_lin(th_range[i], M, d, v, f0)
        a_theta[][i] = columni
        
    a_H= a_theta.conj().T

    Py = np.matmul(a_H, Rx, a_theta)
    return Py

theta0= [0,np.pi/12]
Q = len(theta0)
SNR = 10
th_range = np.linspace(-np.pi/2,np.pi/2, 100)
M=6
d= 1
v= 340
f0= 1000

sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_lin(theta0, M, d, v, f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix
    
    
P_mbf = matchedbeamformer(Rx, th_range, M, d, v, f0)

P_mvdr = mvdr(Rx, th_range, M, d, v, f0)