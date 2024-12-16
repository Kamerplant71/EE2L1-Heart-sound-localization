import numpy as np
import matplotlib.pyplot as plt
import random


def a_lin(theta, M, d, v, f0):
    #Create empty complex vector
    a = (np.ones((M,1),dtype="complex_"))
    for i in range(0,M):
        #Compute each entry for vector
        a[i] = a[i] * np.exp(-1j*(i)*d/v*np.sin(theta)*2*np.pi*f0)
        
    return a

def a_lin_multiplesources(theta, M, d, v, f0):
    a = np.ones((M,len(theta)),dtype="complex_")
    for j in range(0,len(theta)):
        for i in range(0,M):
            a[i][j] = np.exp(-1j*(i)*d/v*np.sin(theta[j])*2*np.pi*f0)
        
    return a

def a_z(s_position,mic_positions, M,v,f0):
    a = (np.ones((M,1),dtype="complex_"))

    for i in range(M):
        a[i]=    1/np.linalg.norm(s_position-mic_positions[i,:])*np.exp(-1j*f0/(2*np.pi)*np.linalg.norm(s_position-mic_positions[i,:])/v)

    return a

def a_z_multiplesources(s_positions, mic_positions, M, v, f0):
    a = np.ones((M,len(s_positions)),dtype="complex_")
    for j in range(0,len(s_positions)):
        for i in range(0,M):
            a[i][j] = 1/np.linalg.norm(s_positions[j,:]-mic_positions[i,:])*np.exp(-1j*f0/(2*np.pi)*np.linalg.norm(s_positions[j,:]-mic_positions[i,:])/v)
        
    return a




mic_positions=np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]])
v=40
f0=150
s_position= [0,0,0]
M=6



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

def mvdr_z(Rx, M, xyz_points, v, f0):
    Py = np.zeros((len(xyz_points)),dtype="complex_")
    Rxinv= np.linalg.inv(Rx)
    
    mic_positions = np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]])
    for i in range(0,len(xyz_points)):
        a = a_z(xyz_points[i,:],mic_positions,M,v,f0)
        a_z_H = a.conj().T
    
        pyd =np.matmul(np.matmul(a_z_H,Rxinv),a)
        py = 1/pyd
        Py[i]= py[0][0]
    
    return Py

x = (np.linspace(0, 20, 1000))
y = (np.linspace(0, 20, 1000))
z = np.linspace(-np.pi/2, np.pi/2, 1000)
xyz_points = np.transpose([x,y,z])
SNR = 10

print(np.shape(xyz_points))
print(len(xyz_points))

s_positions = [[0,0,0],[5,5,0]]

sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_z_multiplesources(s_positions, mic_positions, M, v , f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix
