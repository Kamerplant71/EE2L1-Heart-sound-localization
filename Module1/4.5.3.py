import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft,ifft

# Fs, x = wavfile.read("Module1\hrecording_heart_ownDevice.wav")
# period = 1/Fs
# xn= x/max(abs(x))
# xi = xn**2
# xi = xi + 1e-10
# E=-(xi *np.log10(xi))
# b, a = butter(2, [15/24000], btype='low')
# y = signal.lfilter(b,a,E) #SEE envolope

# Y= fft(y)
# t= np.linspace(0,period*len(y),len(y))
# f= np.linspace(0,Fs, len(Y))
# plt.subplot(211)
# plt.plot(t,y)
# plt.xlabel("Time(s)")
# plt.ylabel("Magnitude")
# plt.title("Time plot SEE")
# plt.xlim(0,3)
# plt.tight_layout()

# plt.subplot(212)
# plt.plot(f,abs(Y))
# plt.xlabel("Frequency(Hz)")
# plt.ylabel("Magnitude")
# plt.title("Frequency plot SEE")
# plt.xlim(0,20)
# plt.show()
# plt.tight_layout()
Fs, x = wavfile.read("Module1\hrecording_heart_ownDevice.wav")

def SEE_plot(Fs,x):
    period = 1/Fs
    xn= x/max(abs(x))
    xi = xn**2
    xi = xi + 1e-10
    E=-(xi *np.log10(xi))
    b, a = butter(2, [15/24000], btype='low')
    y = signal.lfilter(b,a,E) #SEE envolope


    #Y= fft(y)
    t= np.linspace(0,period*len(y),len(y))
    #f= np.linspace(0,Fs, len(Y))

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label="Filtered SEE", alpha=1, color='blue')
    plt.plot(t, E, label="Unfiltered SEE", alpha=0.5, color='grey')

    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.title("Overlay of filtered and unfiltered Shannon energy")
    plt.xlim(0,3)

    plt.legend()

    plt.show()
    plt.tight_layout()


doei=SEE_plot(Fs,x)