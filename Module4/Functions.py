import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
fs = 48000
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

def generate_source(N):
    real = (np.random.randn(round(N)))
    imag = (np.random.randn(round(N)))*1j

    source = real + imag
    source = source/ np.linalg.norm(source)

    return source   
def datamodel(M,N,theta0,d,v,f0):
    A = a_lin_multiplesources(theta0,M,d,v,f0)

    S = np.zeros((len(theta0),N),dtype="complex")
    for i in range(0,len(theta0)):
        s = generate_source(N)
        S[i,:] = s

    X = np.matmul(A,S)
    return X

def music(Rx, Q, M, th_range, d, v, f0):
    U, S , Vh = np.linalg.svd(Rx)
    Un = U[:,Q:M]
    Un_H = Un.conj().T

    Py= np.zeros(len(th_range),dtype='complex')
    for i in range(0,len(th_range)):
        A = a_lin(th_range[i],M,d,v,f0)
        Ah = A.conj().T
        py_denom = np.matmul(np.matmul( np.matmul(Ah,Un),Un_H ),A   )
        py = 1/ py_denom
        Py[i] = py[0][0]
    
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
        Rx = np.cov(X)
        #print(np.shape(Rx))
        Rx_all.append(Rx)
        #print(np.shape(Rx_all))
    Rx_all= np.array(Rx_all)    

    return f_bins, Rx_all

def narrowband_Rx2(signal,nperseg):
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
