import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, TransferFunction, lsim, tf2zpk, lfilter, find_peaks
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
from wavaudioread import wavaudioread
from Functions import powercalculation, plotting, narrowband_Rx2, create_points, music_z

Fs = 48000
period = 1/Fs
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

def cutting(x,Fs,arr_cut):
    arr_Fs = (Fs * arr_cut).astype(int)
    offset = 0
    for cut in arr_Fs:
        start_idx = cut[0] - offset
        end_idx = cut[1] - offset
        x = np.delete(x, slice(start_idx, end_idx), axis=0) 
        # Update the offset based on how many elements were removed
        offset += end_idx - start_idx

    xnorm = x /np.max(x)

    return xnorm


arr_cut = np.array([[0,3.14],[6.73,7.35],[8.6,11],[13.13,14.03]],dtype=float)
audio_matrix2 = cutting(audio_matrix,48000,arr_cut)
xn2= audio_matrix2[:,0]/np.max(abs(audio_matrix2))
xi2 = xn2**2
xi2 = xi2 + 1e-10
E2=-(xi2 *np.log10(xi2))
b, a = butter(2, [15/(Fs/2)], btype='low')
y2 = lfilter(b,a,E2)# SEE envelope
ynorm2= y2/np.max(y2)


peaks, loveisintheair= find_peaks(ynorm2,0.3,distance = Fs/5)
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

print(f"Upper limits: {upperlimitS1}")
print(f"Lower limits: {lowerlimitS1}")
piecesS1 = []
for i in range(len(upperlimitS1)):
    piecesS1.append(audio_matrix2[lowerlimitS1[i]:upperlimitS1[i]])

piecesS1 = np.concatenate(piecesS1)

piecesS2 = []
for i in range(len(upperlimitS2)):
    piecesS2.append(audio_matrix2[lowerlimitS2[i]:upperlimitS2[i]])

piecesS2 = np.concatenate(piecesS2)
print(np.shape(piecesS1))

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
print(np.shape(piecesS2))

nperseg = 400

Q = 2
Mics = 6
d = 0.05
v= 80

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
f_bins, Rx_all, Xall = narrowband_Rx2(piecesS2,nperseg)
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