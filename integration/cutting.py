import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, TransferFunction, lsim, tf2zpk, lfilter, find_peaks
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
from wavaudioread import wavaudioread

# def rms_value(matrix_of_signals):
#     num_columns = matrix_of_signals.shape[1]
#     RMS = np.zeros(num_columns)
#     for i in range(len(matrix_of_signals[0])):
#         RMS[i] = np.sqrt(np.mean(matrix_of_signals[0:,i]**2))
#     max_index = np.argmax(RMS)
#     return RMS, max_index

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
    peaks_old, loveisintheair= signal.find_peaks(ynorm,0.2,distance = Fs/5)
    # plt.plot(t,ynorm)
    # plt.xlim(7,10)
    # plt.show()
    
    diffpeaks_first=[]
    peaksbaby =[]
    for i in range(len(peaks_old)-1):
        diffpeaks_first.append(peaks_old[i+1] - peaks_old[i])

    # print(f"differntpeaks_first:{diffpeaks_first}")
    # print(f"peaks_old:{peaks_old.shape}")
    # av_diff = np.mean(diffpeaks_first)
    # for i in range(len(diffpeaks_first)):
    #     if diffpeaks_first[i]>1:
    #         peaks = np.delete(peaks_old,i,axis=0)
    # print(f"peaks:{peaks.shape}")

    print(f"differntpeaks_first:{diffpeaks_first}")
    print(f"peaks_old:{peaks_old.shape}")

    #   Collect indices to delete
    indices_to_delete = []
    for i in range(len(diffpeaks_first)):
        if diffpeaks_first[i] > 15000:
            indices_to_delete.append(i)

    # Delete all rows in one go
    peaks = np.delete(peaks_old, indices_to_delete, axis=0)

    print(f"peaks:{peaks.shape}")

    diffpeaks=[]
    peaksbaby =[]
    for i in range(len(peaks)-1):
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
    return peaks

Fs = 48000
period = 1/Fs
# Folder containing audio files
folder = "integration/segmentation_sound"

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
peaks, loveisintheair= find_peaks(ynorm,0.2,distance = Fs/5)

upperlimit = []
lowerlimit = []
for i in range(len(peaks)):
    for j in range(len(ynorm)-peaks[i]):
        if ynorm[peaks[i]+j]<0.03:
            upperlimit.append(peaks[i]+j)
            break
    for j in range(peaks[i]): 
        if ynorm[peaks[i]-j]<0.03:
            lowerlimit.append(peaks[i]-j)
            break

print(f"Upper limits: {upperlimit}")
print(f"Lower limits: {lowerlimit}")
plt.subplot(211)
plt.plot(t,ynorm)
plt.xlim(0,5)
for i in range(len(upperlimit)):
    plt.axvline(x=upperlimit[i]/Fs, color='r', linewidth=0.7, linestyle='--')
    plt.axvline(x=lowerlimit[i]/Fs, color='r', linewidth=0.7, linestyle='--')

pieces = []
for i in range(len(upperlimit)):
    pieces.append(audio_matrix[lowerlimit[i]:upperlimit[i]])

pieces = np.concatenate(pieces)

print(pieces)
plt.subplot(212)
plt.plot(pieces)
plt.show()