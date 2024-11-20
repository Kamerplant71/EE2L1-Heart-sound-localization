import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal

from scipy.io import wavfile
from scipy.fft import fft,ifft

Fs, x = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")
b, a = butter(2, [10/2000, 800/2000], btype='band')
g = signal.filtfilt(b, a,x)
f , t , Sxx = spectrogram(x,Fs,nperseg=256)
Sx_dB = 10*np.log10(Sxx)
f2 , t2 , Sxx2 = spectrogram(g,Fs,nperseg=256)
Sx_dB2 = 10*np.log10(Sxx2)

plt.subplot(2,1,1)
plt.pcolormesh(t, f, Sx_dB, shading='gouraud')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")  

plt.subplot(212)
plt.pcolormesh(t2, f2, Sx_dB2, shading='gouraud')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")  
plt.tight_layout()
plt.show()