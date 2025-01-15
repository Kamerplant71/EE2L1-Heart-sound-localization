import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from wavaudioread import wavaudioread
from tkinter import ttk
from scipy.signal import ShortTimeFFT
fs=48000

def powercalculation(Q, Mics, v, x_steps, zmin, zmax, signal, nperseg, fmin, fmax):
    
    
    mic_positions= np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]]) /100
    
    y_steps = x_steps
    xmax = 10 /100
    ymax = 20 /100
    z = np.linspace(zmin,zmax,10)/100
    f_bins, Rx_all, Xall = narrowband_Rx2(signal,nperseg)
    print(f_bins)

    N_Bins = len(f_bins)

    Pytotal = []

    for i in range(len(z)):  # z ranges from 0 to 10
        xyz = create_points(x_steps, y_steps, xmax, ymax, z[i])
        Py_1_layer = np.zeros((x_steps,y_steps))
        # Compute the MUSIC spectrum for the current z-plane
        for i in range(1,len(f_bins)):  # len(f_bins)
            if f_bins[i] >=fmin and f_bins[i] <= fmax:
                Py = music_z(Rx_all[i,:,:],Q,Mics,xyz,v,f_bins[i],mic_positions,x_steps,y_steps)
                #Py = mvdr_z(Rx_all[i,:,:], Mics,xyz,v,f_bins[i],mic_positions,x_steps,y_steps)
                Py_1_layer = Py_1_layer + Py

        print(np.shape(Py_1_layer))
        Pytotal.append(Py_1_layer)
    
    Pytotal = np.array(Pytotal)
    print(np.shape(Pytotal))    
    #pymax = np.max(py)
    return Pytotal, z



# Plot the result
def plotting(Pytotal, z, xmax, ymax, mic_positions):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for i in range(len(z)):
        ax = axes[i]
        im = ax.imshow(np.abs(Pytotal[i,:,:]), extent=(0, xmax * 100, 0, ymax * 100), origin='lower', vmin=0, vmax = np.max(abs(Pytotal)))
        ax.set_title(f"z = {z[i] * 100:.1f} cm")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        mic_x = mic_positions[:, 0] * 100  # Convert to cm
        mic_y = mic_positions[:, 1] * 100  # Convert to cm
        ax.scatter(mic_x, mic_y, color='red', label='Microphones')

    # Add a colorbar to the figure
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    plt.suptitle("MUSIC Spectrum at Different z-Planes", fontsize=16)
    plt.show()
    




def a_z(s_position,mic_positions, M,v,f0):
    a = (np.ones((M,1),dtype="complex_"))

    for i in range(M):
        rm = np.linalg.norm(s_position-mic_positions[i,:])
        a[i]=   1/rm*  np.exp(-1j *f0*2*np.pi * rm /v) 
        #print(rm)


    return a

def a_z_multiplesources(s_positions, mic_positions, M, v, f0):
    a = np.ones((M,len(s_positions)),dtype="complex_")
    for j in range(0,len(s_positions)):
        for i in range(0,M):
            rm = np.linalg.norm(s_positions[j,:]-mic_positions[i,:])
            a[i][j] = 1/rm * np.exp(-1j*f0*2*np.pi* rm/v)
    
    return a


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

def narrowband_Rx2(signal,nperseg):
    Sx_all = []
    win = ('gaussian', 1e-2 * fs) # Gaussian with 0.01 s standard dev.
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap=0,scale_to='magnitude', phase_shift=None)
    f_bins = SFT.f

    for i in range(0,6):            #Go through all mics to calculate Sx
        #if max(signal[:,i]) > 100:  #Only calculate Sx if signal actually has sound

            Mic = signal[:,i]
            Sx = SFT.stft(Mic)
            Sx_all.append(Sx)

    Sx_all = np.array(Sx_all)

    #print(f_bins[N_Bins-1])
    Rx_all = []
    Xall = []
    for j in range(0,len(f_bins)):   #Loop for each frequency bin
        X = Sx_all[:, j, :]
        Rx = np.matmul(X,X.conj().T)/len(signal[:,0]) 
        #print(np.shape(Rx))
        Rx_all.append(Rx)
        Xall.append(X)
        #print(np.shape(Rx_all))4
    Rx_all= np.array(Rx_all)    
    Xall= np.array(Xall)
    return f_bins, Rx_all, Xall