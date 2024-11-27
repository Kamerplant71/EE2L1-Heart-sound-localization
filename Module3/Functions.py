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

def matchedbeamformer(Rx, th_range, M, d, v, f0):
    #Create empty Py
    Py = np.zeros((len(th_range)),dtype="complex_")

    #Loop for each theta
    for i in range(0,len(th_range)):
        #Calculate a_theta
        a_theta= a_lin(th_range[i], M, d, v, f0)
        a_H_theta =a_theta.conj().T

        #Calculate Power
        py= (np.matmul(np.matmul(a_H_theta,Rx),a_theta))
        Py[i] = py[0][0]

    return Py

def mvdr(Rx, th_range, M, d, v, f0):
    Py = np.zeros((len(th_range)),dtype="complex_")
    Rxinv= np.linalg.inv(Rx)

    #Loop for each theta
    for i in range(0,len(th_range)):
        #Calculate a_theta
        a_theta= a_lin(th_range[i], M, d, v, f0)
        a_H_theta = a_theta.conj().T

        #Calculate Power
        py1= (np.matmul(np.matmul(a_H_theta,Rxinv),a_theta))
        py= 1/py1
        Py[i] = py[0][0]

    return Py
   
