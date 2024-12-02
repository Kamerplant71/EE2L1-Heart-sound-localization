import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft,ifft

Fs, x = wavfile.read("Module1\hrecording_heart_ownDevice.wav")
Fs2,x2 = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")
def peaks_baby(Fs,x):
    period = 1/Fs
    xn= x/max(abs(x))
    xi = xn**2
    xi = xi + 1e-10
    E=-(xi *np.log10(xi))

    b, a = butter(2, [15/(Fs/2)], btype='low')
    y = signal.lfilter(b,a,E)# SEE envelope
    y = y.real
    Y= fft(y)
    t= np.linspace(0,period*len(y),len(y))
    f= np.linspace(0,Fs, len(Y))

    pieces = np.array_split(y, 8)
    max_values = [np.max(piece) for piece in pieces]
    mean = np.mean(max_values)
    for i in range(len(max_values)):
        if max_values[i]>1.5*mean:
            pieces[i] = np.zeros(pieces[1]) #deze wat de flip
    
    y_reconstructed = np.concatenate((pieces))
    ynorm = y_reconstructed/max(y_reconstructed)
    peaks, loveisintheair= signal.find_peaks(y_reconstructed,0.2,distance = Fs/5)
    
    plt.plot(t,ynorm)
    plt.xlim(0,3)
    plt.ylim(0,1)
    plt.show()

    peaksbaby =[]
    for i in range(len(peaks)-2):
        dif1 = peaks[i+1] - peaks[i]
        dif2 = peaks[i+2] - peaks[i+1]
        if dif1>dif2:
            peaksbaby.append(("Peak at", peaks[i]/Fs, " seconds is S2"))
        else: peaksbaby.append(("Peak at", peaks[i]/Fs, "seconds is S1"))

        endpeaksbaby=[]
    if len(peaks) >= 3:  
        dif3 = peaks[-2] - peaks[-3] 
        dif4 = peaks[-1] - peaks[-2]  

    if dif3 > dif4:
        endpeaksbaby.append(("Peak at", peaks[-2]/Fs, "is S2"))
        endpeaksbaby.append(("Peak at", peaks[-1]/Fs, "is S1"))
    else:
        endpeaksbaby.append(("Peak at", peaks[-2]/Fs, "is S1"))
        endpeaksbaby.append(("Peak at", peaks[-1]/Fs, "is S2"))


    peaksbaby= np.concatenate((peaksbaby, endpeaksbaby))


    print(peaksbaby)
    return peaksbaby
   
   
hallo = peaks_baby(Fs2,x2)
