import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram, TransferFunction
from scipy import signal


from scipy.io import wavfile
from scipy.fft import fft,ifft
import os
from scipy.signal import butter, spectrogram, TransferFunction, lsim, tf2zpk, lfilter, find_peaks
# from wavaudioread import wavaudioread

def peaks_baby(Fs,x, d, h):
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
    peaks, loveisintheair= signal.find_peaks(ynorm, height = h, distance = d)
    
 
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


def SEE_plot(Fs, x):
    period = 1 / Fs  # Sampling period
    xn = x / np.max(abs(x))  # Normalize the signal
    xi = xn**2
    xi = xi + 1e-10
    E = -(xi * np.log10(xi))  # Shannon energy

    # Design a low-pass filter
    b, a = butter(2, [15 / (Fs / 2)], btype='low')
    y = signal.lfilter(b, a, E)  # Filter the Shannon energy
    ynorm = y / np.max(y)  # Normalize the filtered signal

    # Recalculate the time vector based on the length of the cut signal
    t = np.linspace(0, period * len(y), len(y))

    # Plotting
    plt.close('all') 
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, y, label="Filtered SEE", alpha=1, color='blue')
    plt.plot(t, E, label="Unfiltered SEE", alpha=0.5, color='grey')

    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.title("Overlay of filtered and unfiltered Shannon energy")
    # plt.xlim(0, t[-1])  # Set x-axis limits to the new signal duration

    plt.legend()
    plt.show()
    plt.tight_layout()
    plt.close(fig)

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

    plt.close('all') 
    fig = plt.figure(figsize=(10, 3))

    
    plt.plot(t2,d,label="filtered data", alpha=1, color='blue')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.title("Filtered and unfiltered data")
    

    
    plt.plot(tdata,data, label="unfiltered data", alpha=0.5, color='grey')
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    #plt.title("Prefiltered data")
    plt.ylim(-15000,15000)
    

    plt.legend()

    plt.show()
    plt.tight_layout()
    plt.close(fig)


def plot_spectrogram(Fs, x):
    b, a = butter(2, [10/24000, 800/24000], btype='band')
    g = signal.filtfilt(b, a, x)
    f, t, Sxx = spectrogram(x, Fs, nperseg=256)
    Sx_dB = 10 * np.log10(Sxx)
    f2, t2, Sxx2 = spectrogram(g, Fs, nperseg=256)
    Sx_dB2 = 10 * np.log10(Sxx2)
    
    # Clear all previous plots
    plt.close('all')  
    
    # Create a new figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # First subplot
    axs[0].pcolormesh(t, f, Sx_dB, shading='gouraud')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title("Raw Signal Spectrogram")
    axs[0].colorbar = plt.colorbar(axs[0].pcolormesh(t, f, Sx_dB, shading='gouraud'), ax=axs[0])
    
    # Second subplot
    axs[1].pcolormesh(t2, f2, Sx_dB2, shading='gouraud')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("Filtered Signal Spectrogram")
    axs[1].colorbar = plt.colorbar(axs[1].pcolormesh(t2, f2, Sx_dB2, shading='gouraud'), ax=axs[1])
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def cutting(x,Fs,arr_cut):
    arr_Fs = (Fs * arr_cut).astype(int)
    offset = 0
    for cut in arr_Fs:
        start_idx = cut[0] - offset
        end_idx = cut[1] - offset
        x = np.delete(x, slice(start_idx, end_idx), axis=0) 
        # Update the offset based on how many elements were removed
        offset += end_idx - start_idx

    xnorm = x /np.max(abs(x))

    return xnorm


def cuttingarray(Fs, array_cut_indices, audio_matrix):
    
    arr_Fs = (Fs * array_cut_indices).astype(int)
    offset = 0
    for cut in arr_Fs:
        start_idx = cut[0] - offset
        end_idx = cut[1] - offset
        audio_matrix = np.delete(audio_matrix, slice(start_idx, end_idx), axis=0) 
        # Update the offset based on how many elements were removed
        offset += end_idx - start_idx
    return audio_matrix

def cutting_real1(Fs, audio_matrix, height_threshold,distance_threshold):


    # Print shape of the matrix
    period = 1/Fs
    print("Audio Matrix Shape:", audio_matrix.shape)
    xn= audio_matrix[:,0]/np.max(abs(audio_matrix))
    xi = xn**2
    xi = xi + 1e-10
    E=-(xi *np.log10(xi))
    b, a = butter(2, [15/(Fs/2)], btype='low')
    y = lfilter(b,a,E)# SEE envelope
    ynorm= y/np.max(y)
    t= np.linspace(0,period*len(y),len(y))


    audio_matrix2= audio_matrix	 /np.max(audio_matrix)
    xn2= audio_matrix2[:,0]
    xi2 = xn2**2
    xi2 = xi2 + 1e-10
    E2=-(xi2 *np.log10(xi2))
    b, a = butter(2, [15/(Fs/2)], btype='low')
    y2 = lfilter(b,a,E2)# SEE envelope
    ynorm2= y2/np.max(y2)

    peaks, loveisintheair= find_peaks(ynorm2,height_threshold,distance = distance_threshold)
    diffpeaks=[]
    S1 =[]
    S2 = []
    for i in range(len(peaks)-1):
        diffpeaks.append(peaks[i+1] - peaks[i])
            
    for i in range(len(diffpeaks)-1):
        if diffpeaks[i]>diffpeaks[i+1]:
            S2.append(peaks[i])
        else: S1.append(peaks[i])

    endS1=[]
    endS2=[]
    if len(peaks) >= 3:  
        diffpeaks[-1] = peaks[-2] - peaks[-3] 
        diffpeaks[-2] = peaks[-1] - peaks[-2]  

    if diffpeaks[-1] > diffpeaks[-2]:
        endS1.append(peaks[-2])
        endS2.append(peaks[-1])
    else:
        endS2.append(peaks[-2])
        endS1.append(peaks[-1])


    S1= np.concatenate((S1,endS1))
    S2 = np.concatenate((S2,endS2))

    upperlimitS1 = []
    lowerlimitS1 = []
    upperlimitS2 = []
    lowerlimitS2 = []
    for i in range(len(S1)):
        for j in range(len(ynorm2)-S1[i]):
            if ynorm2[S1[i]+j]<0.03:
                upperlimitS1.append(S1[i]+j)
                break
        for j in range(S1[i]): 
            if ynorm2[S1[i]-j]<0.03:
                lowerlimitS1.append(S1[i]-j)
                break

    for i in range(len(S2)):
        for j in range(len(ynorm2)-S2[i]):
            if ynorm2[S2[i]+j]<0.03:
                upperlimitS2.append(S2[i]+j)
                break
        for j in range(S2[i]): 
            if ynorm2[S2[i]-j]<0.03:
                lowerlimitS2.append(S2[i]-j)
                break

    piecesS1 = []
    for i in range(len(upperlimitS1)):
        piecesS1.append(audio_matrix2[lowerlimitS1[i]:upperlimitS1[i]])

    piecesS1 = np.concatenate(piecesS1)

    piecesS2 = []
    for i in range(len(upperlimitS2)):
        piecesS2.append(audio_matrix2[lowerlimitS2[i]:upperlimitS2[i]])

    piecesS2 = np.concatenate(piecesS2)
    return y, y2, ynorm, ynorm2, piecesS1, piecesS2, upperlimitS1, lowerlimitS1, upperlimitS2, lowerlimitS2

