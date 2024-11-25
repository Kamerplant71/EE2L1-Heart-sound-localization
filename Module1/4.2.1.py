import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal

from scipy.io import wavfile
from scipy.fft import fft,ifft


Fs, data = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")
print(Fs)

b, a = butter(2, [10/24000, 800/24000], btype='band')
dirac = np.concatenate((np.zeros(1500),[1],np.zeros(1500)))
g = signal.filtfilt(b, a,dirac)
G=fft(g)
period = 1/Fs
t1= np.linspace(0,period*len(g),len(g))
f1= np.linspace(0,Fs, len(G))

#d = signal.filtfilt(b, a,data)
d = np.convolve(data,g)
D=fft(d)

t2= np.linspace(0,period*len(d),len(d))
f2= np.linspace(0,Fs, len(D))

plt.subplot(221)
plt.plot(t1,g)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot g")
plt.tight_layout()

plt.subplot(222)
plt.plot(f1,20*np.log10(abs(G)))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency G")
plt.xscale("log")


plt.subplot(223)
plt.plot(t2,d)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot d")
plt.tight_layout()

plt.subplot(111)
plt.plot(f2,abs(D))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency D")
plt.xlim(0,2000)


plt.show()
plt.tight_layout()