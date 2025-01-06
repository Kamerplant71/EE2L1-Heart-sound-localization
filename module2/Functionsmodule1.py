import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft,ifft

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

def Impulse(duration,frequency,relative_amplitude,delay):
    a = 1/duration
    omega = 2*np.pi*frequency
    system = TransferFunction([0,0,omega*relative_amplitude],[1,2*a,a**2+omega**2])
    t, h_almost = signal.impulse(system) # impulse response
    h = np.concatenate((np.zeros(int(delay*Fs)),h_almost))
    return h

def zero_pad_arrays(t1,t2,t3,t4):
    mlen = max(map(len, [t1, t2, t3,t4]))
    t1_new = np.concatenate((t1,np.zeros(mlen-len(t1))))
    t2_new = np.concatenate((t2,np.zeros(mlen-len(t2))))
    t3_new = np.concatenate((t3,np.zeros(mlen-len(t3))))
    t4_new = np.concatenate((t4,np.zeros(mlen-len(t4))))
    return t1_new, t2_new, t3_new, t4_new

def h_combine(h1,h2,h3,h4):
    htotal = h1+h2+h3+h4
    Htotal = fft(htotal)
    t = np.linspace(0,len(htotal)/Fs,len(htotal))
    w = np.linspace(0,2*np.pi*Fs,len(Htotal))
    return htotal, Htotal, t, w

def input_signal(Ns,list_of_impulse_responses):
    s = [None]*len(list_of_impulse_responses)
    x = [None]*len(list_of_impulse_responses)
    for i in range(len(list_of_impulse_responses)):

        s[i]=np.concatenate((np.random.randn(Ns),np.zeros(len(list_of_impulse_responses[i]-Ns))))
        x[i]=np.convolve(s[i],list_of_impulse_responses[i],mode = "full")
    return x

def bpm_more_signals(Matrix_of_signals, BPM):
    BPS = BPM / 60  # Convert BPM to BPS
     # Initialize a matrix with zeros
    X = np.zeros((6, (len(Matrix_of_signals[0]) + int(BPS * Fs))*10))
    
    for i in range(len(Matrix_of_signals)):
        # Concatenate and tile signals to replace zeros
        signal_extension = np.tile(np.concatenate((Matrix_of_signals[i], np.zeros(int(BPS * Fs)))), 10)
        # Update row i in matrix X
        X[i] = signal_extension  # Ensure dimensions match

    return X

def ch3(x,y,epsi):
    Nx = len(x)            # Length of x
    Ny = len(y)             # Length of y
    L= Ny - Nx +1            # Length of h
    print(Nx)
    print(Ny)
    print(L)

    # Force x to be the same length as y
    x = np.append(x, [0]* (L-1))

    print(len(x))
    print(len(y))
    # Deconvolution in frequency domain
    Y = fft(y)
    X = fft(x)
    H = Y/X

    # Threshold to avoid blow ups of noise during inversion
    for i in range(0,L-1):
      if abs(X[i]) < epsi*max(np.absolute(X)):
        H[i] = 0

    h = np.real(ifft(H))   # ensure the result is real
    h = h[0:L]    # optional: truncate to length Lhat (L is not reliable?)
    return h