import numpy as np
import matplotlib.pyplot as plt

def a_z(s_position,mic_positions, M,v,f0):
    a = (np.ones((M,1),dtype="complex_"))

    for i in range(M):
        rm = np.linalg.norm(s_position-mic_positions[i,:])
        a[i]=    1/rm*  np.exp(-1j *f0*2*np.pi * rm /v)
        #print(rm)


    return a

def a_z_multiplesources(s_positions, mic_positions, M, v, f0):
    a = np.ones((M,len(s_positions)),dtype="complex_")
    for j in range(0,len(s_positions)):
        for i in range(0,M):
            rm = np.linalg.norm(s_positions[j,:]-mic_positions[i,:])
            a[i][j] = 1/rm * np.exp(-1j*f0*2*np.pi* rm/v)
    
    return a

def mvdr_z(Rx, M, xyz_points, v, f0, mic_positions,xsteps,ysteps):
    Py = np.zeros((xsteps* ysteps,1),dtype="complex_")
    Rxinv= np.linalg.inv(Rx)
    

    for i in range(0,len(xyz_points)):
        a = a_z(xyz_points[i,:],mic_positions,M,v,f0)
#        print(xyz_points[i,:])
        a_z_H = a.conj().T
    
        pyd =np.matmul(np.matmul(a_z_H,Rxinv),a)
        py = 1/pyd
        Py[i]= py[0][0]

    Py = np.reshape(Py,(x_steps,y_steps))

    return Py



def create_points(x_steps,y_steps, x_max, y_max, z):
    x = np.linspace(0,x_max,x_steps)
    y = np.linspace(0,y_max,y_steps)

    X,Y = np.meshgrid(x,y)
    total_steps = x_steps*y_steps

    X = np.reshape(X,(total_steps,1))
    Y = np.reshape(Y,(total_steps,1))
    Z = z * np.ones((total_steps,1))
    xyz =  np.append( np.append(X,Y,axis=1),Z  , axis=1)

    return xyz



SNR = 10
mic_positions= np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]]) /100

v=60
f0= 150

s_position= np.array([10,10,10]) / 100

#s_positions = np.array([[10,10,3],[10,10,6]])

M=6



sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_z(s_position, mic_positions, M, v , f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix


x_steps = y_steps = 200
xmax = ymax = 20 /100
z = 10 / 100


xyz = create_points(x_steps,y_steps,xmax,ymax,z)
print(xyz)
p = mvdr_z(Rx, M, xyz, v, f0, mic_positions,x_steps,y_steps)

# Plot using imshow
plt.imshow(abs(p), extent=(0, xmax*100, 0, ymax*100), origin='lower', aspect='auto')
plt.colorbar(label="Power")
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('MVDR Power Pattern')
plt.show()