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
        rm = np.linalg.norm(s_position-mic_positions[i,:])
        a[i]=     np.exp(-1j *f0*2*np.pi * rm /v)

    return a

def a_z_multiplesources(s_positions, mic_positions, M, v, f0):
    a = np.ones((M,len(s_positions)),dtype="complex_")
    for j in range(0,len(s_positions)):
        for i in range(0,M):
            rm = np.linalg.norm(s_positions[j,:]-mic_positions[i,:])
            a[i][j] = 1/rm * np.exp(-1j*f0*2*np.pi* rm/v)
        
    return a



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
    Py = np.zeros((40000,1),dtype="complex_")
    Rxinv= np.linalg.inv(Rx)
    
    mic_positions = np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]])
    for i in range(0,len(xyz_points)):
        a = a_z(xyz_points[i,:],mic_positions,M,v,f0)
#        print(xyz_points[i,:])
        a_z_H = a.conj().T
    
        pyd =np.matmul(np.matmul(a_z_H,Rxinv),a)
        py = 1/pyd
        Py[i]= py[0][0]



    return Py


SNR = 10
mic_positions=np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]])
v=60
f0=50
s_position= [10,10,3]
s_positions = np.array([[10,10,3],[10,10,6]])

M=6

nx, ny = (200,200)
x = np.linspace(0,20,200)
y = np.linspace(0,20,200)
xv,yv = np.meshgrid(x,y)


#print(xv)
#print(yv)

xv = np.reshape(xv, (40000,1))
yv = np.reshape(yv, (40000,1) )
z = 3 *  np.ones((40000,1))
#print(xv)
#print(np.shape(xv))

xy = np.append(xv,yv,axis=1)
xyz = np.append(xy,z,axis=1)


#print(np.shape(xyz_points))
#print(len(xyz_points))


sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_z(s_position, mic_positions, M, v , f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix


p = mvdr_z(Rx, M, xyz, v, f0)


grid = abs(p).reshape((ny, nx))


# Plot using imshow
plt.imshow(grid, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', aspect='auto')
plt.colorbar(label="Power")
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('MVDR Power Pattern')
plt.show()