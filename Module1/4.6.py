import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft,ifft

Fs, x = wavfile.read("Module1\hrecording_heart_ownDevice.wav")
period = 1/Fs
xn= x/max(abs(x))
xi = xn**2
xi = xi + 1e-10
E=-(xi *np.log10(xi))

b, a = butter(2, [15/24000], btype='low')
y = signal.lfilter(b,a,E)# SEE envelope
y = y.real
Y= fft(y)
t= np.linspace(0,period*len(y),len(y))
f= np.linspace(0,Fs, len(Y))

Ynorm = Y/max(Y)
ynorm = y/max(y)




peaks, loveisintheair= signal.find_peaks(ynorm,0.4,distance = 9600)
print(peaks)
plt.plot(t,ynorm)
plt.xlim(0,3)
plt.ylim(0,1)
plt.show()
peaksbaby =[]
for i in range(len(peaks)-2):
    dif1 = peaks[i+1] - peaks[i]
    dif2 = peaks[i+2] - peaks[i+1]
    if dif1>dif2:
        peaksbaby.append(("Peak at", peaks[i], "is S2"))
    else: peaksbaby.append(("Peak at", peaks[i], "is S1"))
print(peaksbaby)