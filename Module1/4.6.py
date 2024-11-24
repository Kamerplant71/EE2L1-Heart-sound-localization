import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft,ifft

Fs, x = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")
period = 1/Fs
xn= x/max(abs(x))
xi = xn**2
xi = xi + 1e-10
E=-(xi *np.log10(xi))

b, a = butter(2, [15/2000], btype='low')
y = signal.lfilter(b,a,E)# SEE envolope
y = y.real
Y= fft(y)
t= np.linspace(0,period*len(y),len(y))
f= np.linspace(0,Fs, len(Y))

ynorm = y/max(abs(y))

peaks, loveisintheair= signal.find_peaks(x, height=0.1, distance=1/Fs * 900)



