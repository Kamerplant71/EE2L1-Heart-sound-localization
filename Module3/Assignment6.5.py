import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from wavaudioread import wavaudioread
from Functions import mvdr, matchedbeamformer


def narrowband_Rx(signal,nperseg):
    Sx_all = []
    win = ('gaussian', 1e-2 * fs) # Gaussian with 0.01 s standard dev.
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap=0,scale_to='magnitude', phase_shift=None)
    f_bins = SFT.f

    for i in range(0,8):            #Go through all mics to calculate Sx
        if max(signal[:,i]) > 100:  #Only calculate Sx if signal actually has sound

            Mic = signal[:,i]
            Sx = SFT.stft(Mic)
            Sx_all.append(Sx)

    Sx_all = np.array(Sx_all)

    #print(f_bins[N_Bins-1])
    Rx_all = []
    for j in range(0,len(f_bins)):   #Loop for each frequency bin
        X = Sx_all[:, j, :]
        Rx = np.cov(X)
        #print(np.shape(Rx))
        Rx_all.append(Rx)
        #print(np.shape(Rx_all))
    Rx_all= np.array(Rx_all)    

    return f_bins, Rx_all

fs = 48000
#signal = wavaudioread("Module3\Recordings\d1source_60degrees.wav",fs)
signal1 = wavaudioread("Module3\Recordings\d1source_0degrees.wav",fs)
signal2 = wavaudioread("Module3\Recordings\d1source_20degrees.wav",fs)
signal = signal1[:623040,] + signal2
th_range = np.linspace(-np.pi/2,np.pi/2, 1000)
nperseg = 100
N_Bins = 50

Mics = 6
d = 0.1
v=340

f_bins, Rx_all = narrowband_Rx(signal,nperseg)

Pytotal = np.zeros(len(th_range))

#Averaging the power in case of a real signal
for i in range(1,len(f_bins)):
    Py = matchedbeamformer(Rx_all[i,:,:],th_range,Mics,d,v,f_bins[i])
    Pytotal = Pytotal + Py
PyAvg_mbf = Pytotal / len(f_bins)

Pytotal = np.zeros(len(th_range))

#Averaging the power in case of a real signal
for i in range(1,len(f_bins)):
    Py = mvdr(Rx_all[i,:,:],th_range,Mics,d,v,f_bins[i])
    Pytotal = Pytotal + Py
PyAvg_mvdr = Pytotal / len(f_bins)

print(len(f_bins))
print(f_bins[1])
plt.rcParams['figure.figsize'] = [7 , 5]
plt.rcParams['figure.dpi'] = 150


plt.plot(th_range*180/np.pi,abs(PyAvg_mbf)/max(abs(PyAvg_mbf)))
plt.xlabel("Angle [deg]", fontsize = 20)
plt.ylabel("Normalized Power", fontsize =20)
plt.title("Spatial spectrum, Matched Beamformer", fontsize = 20) 
plt.vlines(0,ymin=0,ymax=1,linestyles="dashed",colors="r")
plt.vlines(20,ymin=0,ymax=1,linestyles="dashed",colors="r")
plt.legend(["Power", "Expected angle"],loc= "upper left", fontsize = 16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

#Peak angle
for i in range(len(th_range)):
    if PyAvg_mbf[i] == max(PyAvg_mbf):
        print(th_range[i]*180/np.pi)

plt.tight_layout()
plt.show()

plt.plot(th_range*180/np.pi,abs(PyAvg_mvdr)/max(abs(PyAvg_mvdr)))
plt.xlabel("Angle [deg]",fontsize=20)
plt.ylabel("Normalized Power",fontsize= 20)
plt.title("Spatial spectrum, MVDR",fontsize =20 ) 
plt.vlines(0,ymin=0,ymax=1,linestyles="dashed",colors="r")
plt.vlines(20,ymin=0,ymax=1,linestyles="dashed",colors="r")
plt.legend(["Power", "Expected angle"],loc= "upper left", fontsize = 16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
#Peak angle
for i in range(len(th_range)):
    if PyAvg_mvdr[i] == max(PyAvg_mvdr):
        print(th_range[i]*180/np.pi)



plt.tight_layout()
plt.show()