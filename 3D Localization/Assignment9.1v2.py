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

    Py = np.reshape(Py,(xsteps,ysteps))

    return Py


def music_z(Rx, Q, M, xyz_points, v, f0, mic_positions, xsteps, ysteps):
    U, S , V_H = np.linalg.svd(Rx)
    Un = U[:,Q:M]
    Un_H = Un.conj().T

    Py = np.zeros((xsteps*ysteps,1),dtype = "complex")
    
    for i in range(0,len(xyz_points)):
        a = a_z(xyz_points[i,:],mic_positions,M,v,f0)
#       print(xyz_points[i,:])
        a_z_H = a.conj().T
    
        pyd =np.matmul(np.matmul(np.matmul(a_z_H, Un),Un_H),a)
        py = 1/pyd
        Py[i]= py[0][0]

    Py = np.reshape(Py,(xsteps,ysteps))

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

#s_position= np.array([10,5,15]) / 100

s_position = np.array([[5, 7.5 ,12],[5,15,18]]) /100

M=6



sigma_n = 10**(-SNR/20); # std of the noise (SNR in dB)
A = a_z_multiplesources(s_position, mic_positions, M, v , f0); # source direction vectors
A_H = A.conj().T
R = A @ A_H # assume equal powered sources
Rn = np.eye(M,M)*sigma_n**2; # noise covariance
Rx = R + Rn # received data covariance matrix


x_steps = y_steps = 100
xmax = 10 /100
ymax = 20 /100
z = 10 / 100
Q=2


xyz = create_points(x_steps,y_steps,xmax,ymax,z)
print(xyz)
#p = music_z(Rx, Q, M, xyz, v, f0, mic_positions,x_steps,y_steps)


'''
# Plot using imshow
plt.imshow(abs(p), extent=(0, xmax*100, 0, ymax*100), origin='lower', aspect='auto')
plt.colorbar(label="Power")
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('MVDR Power Pattern')
plt.show()

'''
fig, axes = plt.subplots(2, 5, figsize=(15, 8), constrained_layout=True)
axes = axes.flatten()
py = []

z = np.linspace(11,20,10)/100


for i in range(len(z)):  # z ranges from 0 to 10
    xyz = create_points(x_steps, y_steps, xmax, ymax, z[i])
    
    # Compute the MUSIC spectrum for the current z-plane
    p = music_z(Rx, Q, M, xyz, v, f0, mic_positions, x_steps, y_steps)
    p = mvdr_z(Rx, M, xyz, v, f0, mic_positions, x_steps, y_steps)
    print(np.shape(p))
    py.append(p)
py = np.array(py)
print(np.shape(py))    
pymax = np.max(py)
    # Plot the result
    
for i in range(len(z)):
    ax = axes[i]
    im = ax.imshow(np.abs(py[i,:,:])/abs(pymax), extent=(0, xmax * 100, 0, ymax * 100), vmin =0, vmax = 1, origin='lower')
    ax.set_title(f"z = {z[i] * 100:.1f} cm")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

# Add a colorbar to the figure
fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
plt.suptitle("MUSIC Spectrum at Different z-Planes", fontsize=16)
plt.show()