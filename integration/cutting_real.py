import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, TransferFunction, lsim, tf2zpk, lfilter, find_peaks
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
from wavaudioread import wavaudioread
Fs = 48000
period = 1/Fs
folder = "integration/segmentation_sound"
arr_cut = np.array([[0,3.14],[6.73,7.35],[8.6,11],[13.13,14.03]],dtype=float)

def cutting_real(Fs, folder, array_cut_indices,height_threshold,distance_threshold):

    # Get list of all .wav files in the folder
    wav_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.WAV')]

    # Initialize an empty list to store signals
    audio_signals = []

    # Read each .wav file
    for wav_file in wav_files:
        signal = wavaudioread(wav_file, Fs)  # Read file
        audio_signals.append(signal)

    # Convert to matrix (numpy array)
    audio_matrix = np.array(audio_signals).T

    # Print shape of the matrix
    print("Audio Matrix Shape:", audio_matrix.shape)
    xn= audio_matrix[:,0]/np.max(abs(audio_matrix))
    xi = xn**2
    xi = xi + 1e-10
    E=-(xi *np.log10(xi))
    b, a = butter(2, [15/(Fs/2)], btype='low')
    y = lfilter(b,a,E)# SEE envelope
    ynorm= y/np.max(y)
    t= np.linspace(0,period*len(y),len(y))

    arr_Fs = (Fs * array_cut_indices).astype(int)
    offset = 0
    for cut in arr_Fs:
        start_idx = cut[0] - offset
        end_idx = cut[1] - offset
        audio_matrix = np.delete(audio_matrix, slice(start_idx, end_idx), axis=0) 
        # Update the offset based on how many elements were removed
        offset += end_idx - start_idx

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

y, y2, ynorm , ynorm2, piecesS1, piecesS2, upperlimitS1,  lowerlimitS1, upperlimitS2, lowerlimitS2= cutting_real(Fs, folder, arr_cut,0.3,Fs/5)
plt.subplot(411)
t= np.linspace(0,period*len(y),len(y))
plt.plot(t,ynorm)

plt.subplot(412)
t2= np.linspace(0,period*len(y2),len(y2))
for i in range(len(upperlimitS1)):
    plt.axvline(x=upperlimitS1[i]/Fs, color='r', linewidth=0.7, linestyle='--')
    plt.axvline(x=lowerlimitS1[i]/Fs, color='r', linewidth=0.7, linestyle='--')
for i in range(len(upperlimitS2)):
    plt.axvline(x=upperlimitS2[i]/Fs, color='b', linewidth=0.7, linestyle='--')
    plt.axvline(x=lowerlimitS2[i]/Fs, color='b', linewidth=0.7, linestyle='--')
plt.plot(t2,ynorm2)

plt.subplot(413)
t3 = np.linspace(0,period*len(piecesS1),len(piecesS1)) 
plt.plot(t3, piecesS1)

plt.subplot(414)
t4 = np.linspace(0,period*len(piecesS2),len(piecesS2)) 
plt.plot(t4,piecesS2)
# plt.plot(audio_matrix2)
plt.show()

