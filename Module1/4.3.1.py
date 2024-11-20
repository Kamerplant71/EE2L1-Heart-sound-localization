import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft


Fs, data = wavfile.read("module1\LinearArray-0-degrees\hrecording_2024-09-30_12-48-30_channel_1.wav")
b, a = butter(2, [10/24000, 800/24000], btype='band')
dirac = np.concatenate((np.zeros(1500),[1],np.zeros(1500)))
g = signal.filtfilt(b, a,data)
g_new = g[::12]
G = fft(g)
G_new=fft(g_new)
period = 1/Fs
period2= 1/4000

t1= np.linspace(0,period*len(g),len(g))
f1= np.linspace(0,Fs, len(G))

t2= np.linspace(0,period2*len(g_new),len(g_new))
f2= np.linspace(0,4000, len(G_new))

plt.subplot(211)
plt.plot(t1,g)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot g")
plt.tight_layout()

plt.subplot(212)
plt.plot(f1,20*np.log10(abs(G)))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency G")
plt.xscale("log")

plt.subplot(211)
plt.plot(t2,g_new)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot g")
plt.tight_layout()

plt.subplot(212)
plt.plot(f2,20*np.log10(abs(G_new)))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency G")
plt.xscale("log")



plt.show()
plt.tight_layout()