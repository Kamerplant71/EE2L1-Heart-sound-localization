import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from scipy import signal

from scipy.io import wavfile
from scipy.fft import fft,ifft


Fs, data = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")
print(Fs)
def Data_plot(Fs,data):
    b, a = butter(2, [10/24000, 800/24000], btype='band')
    dirac = np.concatenate((np.zeros(1500),[1],np.zeros(1500)))
    g = signal.filtfilt(b, a,dirac)
    G =fft(g)
    period = 1/Fs
    t1= np.linspace(0,period*len(g),len(g))
    f1= np.linspace(0,Fs, len(G))

    tdata = np.linspace(0,period*len(data),len(data))
    d = np.convolve(data,g)
    d = signal.filtfilt(b, a,data)
    D=fft(d)

    t2= np.linspace(0,period*len(d),len(d))
    f2= np.linspace(0,Fs, len(D))


    plt.figure(figsize=(10, 3))

    plt.plot()
    plt.plot(t2,d,label="filtered data", alpha=1, color='blue')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.title("Filtered and unfiltered data")
    plt.tight_layout()

    plt.plot()
    plt.plot(tdata,data, label="unfiltered data", alpha=0.5, color='grey')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    #plt.title("Prefiltered data")
    plt.ylim(-15000,15000)
    plt.tight_layout()

    plt.legend()

    plt.show()
    plt.tight_layout()

John=Data_plot(Fs,data)