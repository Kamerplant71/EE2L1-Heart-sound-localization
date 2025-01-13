import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, TransferFunction, lsim, tf2zpk
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
from wavaudioread import wavaudioread

def ch3(x,y,Lhat, epsi):
    Nx = len(x)            # Length of x
    Ny = len(y)             # Length of y
    L= Ny - Nx +1            # Length of h
    print(Nx)
    print(Ny)
    print(L)

    # Force x to be the same length as y
    x = np.append(x, [0]* (L-1))

    print(len(x))
    print(len(y))
    # Deconvolution in frequency domain
    Y = fft(y)
    X = fft(x)
    H = Y/X

    # Threshold to avoid blow ups of noise during inversion
    for i in range(0,L-1):
      if abs(X[i]) < epsi*max(np.absolute(X)):
        H[i] = 0

    h = np.real(ifft(H))   # ensure the result is real
    h = h[0:Lhat]    # optional: truncate to length Lhat (L is not reliable?)
    return h

Fs, source = wavfile.read("module2/recording_2024-12-18_11-50-28_channel_0_source.wav")
Fs_m, microphone = wavfile.read("module2/recording_2024-12-18_11-50-28_channel_1_microphone.wav")

b, a = butter(2, [300/24000], btype='high')
d, c = butter(2, [300/24000], btype='high')

h = ch3(source ,microphone,48000,0.0005)
d=0.22 #meters
#v=d/t 
print(h)
signal.correlate()


period=1/Fs

t1= np.linspace(0,period*len(h),len(h))
 
plt.plot(t1,h)
plt.xlim(0,0.05)
plt.show()