import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from wavaudioread import wavaudioread

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

def mvdr_z2(Rx, M, xyz_points, v, f0, mic_positions,length):
    Py = np.zeros((length),dtype="complex_")
    Rxinv= np.linalg.inv(Rx)
    

    for i in range(0,len(xyz_points)):
        a = a_z(xyz_points[i,:],mic_positions,M,v,f0)
#        print(xyz_points[i,:])
        a_z_H = a.conj().T
    
        pyd =np.matmul(np.matmul(a_z_H,Rxinv),a)
        py = 1/pyd
        Py[i]= py[0][0]

#    Py = np.reshape(Py,(x_steps,y_steps))

    return Py

def narrowband_Rx(signal,nperseg):
    Sx_all = []
    win = ('gaussian', 1e-2 * fs) # Gaussian with 0.01 s standard dev.
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap=0,scale_to='magnitude', phase_shift=None)
    f_bins = SFT.f

    for i in range(0,8):            #Go through all mics to calculate Sx
        if max(signal[:,i]) > 100:  #Only calculate Sx if signal actually has sound

            Mic = signal[:,i]
            Sx = SFT.stft(Mic)
            Sx_all.append(Sx)

    Sx_all = np.array(Sx_all)

    #print(f_bins[N_Bins-1])
    Rx_all = []
    for j in range(0,len(f_bins)):   #Loop for each frequency bin
        X = Sx_all[:, j, :]
        Rx = np.cov(X)/len(signal[:,0])
        #print(np.shape(Rx))
        Rx_all.append(Rx)
        #print(np.shape(Rx_all))
    Rx_all= np.array(Rx_all)    

    return f_bins, Rx_all

def conversion_th_xyz(th_range,r):
    length = len(th_range)
    x = []
    y = []
    for i in range(length):
        x.append(r*np.cos(th_range[i]+np.pi/2))
        y.append(r*np.sin(th_range[i]+np.pi/2))

    X = np.reshape(x,(length,1))
    Y = np.reshape(y,(length,1))
    Z = np.zeros((length,1))

    xyz =  np.append( np.append(X,Y,axis=1),Z  , axis=1)

    return xyz

fs = 48000
signal1 = wavaudioread("Module3\Recordings\d1source_0degrees.wav",fs)
signal2 = wavaudioread("Module3\Recordings\d1source_20degrees.wav",fs)
signal = signal1[:623040,] + signal2

nperseg = 200
Mics = 6
d = 0.1
v=340
mic_positions =   np.array([[-0.25,0,0],
                            [-0.15,0,0],
                            [-0.05,0,0],
                            [ 0.05,0,0],
                            [ 0.15,0,0],
                            [ 0.25,0,0]] )

th_range= np.linspace(-np.pi/2,np.pi/2, 1000)
xyz = conversion_th_xyz(th_range,8)
f_bins, Rx_all = narrowband_Rx(signal,nperseg)


Pytotal = np.zeros(len(th_range))

for i in range(1,len(f_bins)):
    Py = mvdr_z2(Rx_all[i,:,:],Mics,xyz,v,f_bins[i],mic_positions,len(th_range))
    Pytotal = Pytotal + Py

Py_norm = Pytotal /max(Pytotal)

plt.plot(th_range*180/np.pi,abs(Py_norm))
plt.xlabel("Angle [deg]")
plt.ylabel("Normalized Power")
plt.title("Spatial spectrum, Matched Beamformer") 
#plt.vlines(0,ymin=0,ymax=1,linestyles="dashed",colors="r")
#plt.legend(["Power", "Expected angle"],loc= "upper left", fontsize = 16)
#plt.yticks(fontsize=16)
#plt.xticks(fontsize=16)
plt.show()

