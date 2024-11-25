import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal

from scipy.io import wavfile
from scipy.fft import fft,ifft

Fs, x = wavfile.read("Module1\hrecording_heart_ownDevice.wav")
b, a = butter(2, [10/24000, 800/24000], btype='band')
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