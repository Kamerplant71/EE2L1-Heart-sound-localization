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
            pieces[i] = np.zeros(len(pieces[1]))
    
    y_reconstructed = np.concatenate((pieces))
    ynorm = y_reconstructed/max(y_reconstructed)
    peaks, loveisintheair= signal.find_peaks(ynorm,0.2,distance = Fs/5)

    # plt.plot(t,ynorm)
    # plt.xlim(7,10)
    # plt.show()
    # diffpeaksS1=[]
    # diffpeaksS2=[]
    diffpeaks=[]
    peaksbaby =[]
    for i in range(len(peaks)-1):
        print(i)
        diffpeaks.append(peaks[i+1] - peaks[i])

    for i in range(len(diffpeaks)-1):
        if diffpeaks[i]>diffpeaks[i+1]:
            peaksbaby.append(("Peak at", np.round(peaks[i]/Fs,2), "seconds is S2"))
        else: peaksbaby.append(("Peak at", np.round(peaks[i]/Fs,2), "seconds is S1"))

    endpeaksbaby=[]
    if len(peaks) >= 3:  
        diffpeaks[-1] = peaks[-2] - peaks[-3] 
        diffpeaks[-2] = peaks[-1] - peaks[-2]  

    if diffpeaks[-1] > diffpeaks[-2]:
        endpeaksbaby.append(("Peak at", np.round(peaks[-2]/Fs,2), "seconds is S2"))
        endpeaksbaby.append(("Peak at", np.round(peaks[-1]/Fs,2), "seconds is S1"))
    else:
        endpeaksbaby.append(("Peak at", np.round(peaks[-2]/Fs,2), "seconds is S1"))
        endpeaksbaby.append(("Peak at", np.round(peaks[-1]/Fs,2), "seconds is S2"))


    peaksbaby= np.concatenate((peaksbaby, endpeaksbaby))

    # Plot the signal and annotate peaks
    plt.figure(figsize=(12, 6))
    plt.plot(t, ynorm)  # Plot without label
    plt.scatter(t[peaks], ynorm[peaks], color="red")  # Plot peaks without label

    # Annotate each peak with only 'S1' or 'S2'
    for peak_info, peak_idx in zip(peaksbaby, peaks):
        label = peak_info[-1].split()[-1]  # Extract only 'S1' or 'S2'
        plt.text(t[peak_idx], ynorm[peak_idx] + 0.02, label, fontsize=9, color="blue", ha="center")

    #plt.xlim(0, 10)
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Amplitude", fontsize=16)
    plt.title("Heartsound peaks with S1 and S2")
    plt.grid(True)  # Optional: Keep the grid for better visualization
    plt.show()
    #print(peaksbaby)
    return peaksbaby
   
   
hallo = peaks_baby(Fs2,x2)
