import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, TransferFunction, lsim, tf2zpk, lfilter, find_peaks
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
from wavaudioread import wavaudioread
from Functions import powercalculation, plotting, narrowband_Rx2, create_points, music_z

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

x = wavaudioread("recordings\pos2_S1_recording.wav", Fs)

xn= x[:,0]/np.max(abs(x))
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
plt.subplot(311)
plt.plot(t,ynorm)
plt.xlim(10,15)
for i in range(len(upperlimit)):
    plt.axvline(x=upperlimit[i]/Fs, color='r', linewidth=0.7, linestyle='--')
    plt.axvline(x=lowerlimit[i]/Fs, color='r', linewidth=0.7, linestyle='--')

plt.subplot(312)
plt.plot(t,x)
plt.xlim(10,15)
for i in range(len(upperlimit)):
    plt.axvline(x=upperlimit[i]/Fs, color='r', linewidth=0.7, linestyle='--')
    plt.axvline(x=lowerlimit[i]/Fs, color='r', linewidth=0.7, linestyle='--')

pieces = []
for i in range(len(upperlimit)):
    pieces.append(x[lowerlimit[i]:upperlimit[i]])

pieces = np.concatenate(pieces)

print(pieces)
plt.subplot(313)
plt.plot(pieces)
plt.show()

nperseg = 400

Q = 2
Mics = 6
d = 0.05
v=200

x_steps = y_steps = 50
xmax = 10 /100
ymax = 20 /100
zmin= 4
zmax = 15

z = np.linspace(zmin,zmax,10)/100
fmin = 0
fmax = 480
mic_positions= np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]]) /100

"""
Pytotal,z = powercalculation(Q,Mics,v,x_steps,zmin,zmax, pieces,nperseg,fmin,fmax)
plotting(Pytotal,z,xmax,ymax,mic_positions)
"""
f_bins, Rx_all, Xall = narrowband_Rx2(pieces,nperseg)
print(f_bins)

N_Bins = len(f_bins)

Pytotal = []

for j in range(N_Bins):
    if (j-1)%2 == 0:
        fmin = f_bins[j]
        fmax = f_bins[j+2]
        print(fmin,fmax)
        Pytotal= []
        for i in range(len(z)):  # z ranges from 0 to 10
            xyz = create_points(x_steps, y_steps, xmax, ymax, z[i])
            Py_1_layer = np.zeros((x_steps,y_steps))
            # Compute the MUSIC spectrum for the current z-plane
            for i in range(1,len(f_bins)): 
                if f_bins[i] >=fmin and f_bins[i] <= fmax:
                    Py = music_z(Rx_all[i,:,:],Q,Mics,xyz,v,f_bins[i],mic_positions,x_steps,y_steps)
                    #Py = mvdr_z(Rx_all[i,:,:],Mics,xyz,v,f_bins[i],mic_positions,x_steps,y_steps)
                    Py_1_layer = Py_1_layer + Py

            #print(np.shape(Py_1_layer))
            Pytotal.append(Py_1_layer)

        Pytotal = np.array(Pytotal)
        #print(np.shape(Pytotal))    
        #pymax = np.max(py)


        # Plot the result

        fig, axes = plt.subplots(2, 5, figsize=(15, 8), constrained_layout=True)
        axes = axes.flatten()

        for i in range(len(z)):
            ax = axes[i]
            im = ax.imshow(np.abs(Pytotal[i,:,:]), extent=(0, xmax * 100, 0, ymax * 100), origin='lower', vmin=0, vmax = np.max(abs(Pytotal)))
            ax.set_title(f"z = {z[i] * 100:.1f} cm")
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            mic_x = mic_positions[:, 0] * 100  # Convert to cm
            mic_y = mic_positions[:, 1] * 100  # Convert to cm
            ax.scatter(mic_x, mic_y, color='red', label='Microphones')

        # Add a colorbar to the figure
        fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
        plt.suptitle(f"MUSIC Spectrum at Different z-Planes for frequency two source {fmin} till {fmax} Hz", fontsize=16)
        fig.savefig(f"P1_S1_v_{v}_npserseg_200_frequency_{fmin}_till_{fmax}.png", dpi=300)