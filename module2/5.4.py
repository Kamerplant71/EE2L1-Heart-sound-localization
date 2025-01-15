import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, TransferFunction, lsim, tf2zpk
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
from wavaudioread import wavaudioread

def ch3(x,y,epsi):
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
    h = h[0:L]    # optional: truncate to length Lhat (L is not reliable?)
    return h

Fs = 48000
signal = wavaudioread("module2/240610_143941/TRACK01.WAV",Fs)
input = wavaudioread("module2/white_noise_2_channel.wav",Fs)
print(np.shape(signal))
print(np.shape(input))
h = ch3(input,signal,0.01)


plt.plot(h)
plt.show()