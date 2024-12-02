import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from wavaudioread import wavaudioread
from Functions import mvdr, matchedbeamformer


def narrowband_Rx(signal,nperseg,N_Bins):
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
    for j in range(0,N_Bins):   #Loop for each frequency bin
        X = Sx_all[:, j, :]
        Rx = np.cov(X)
        #print(np.shape(Rx))
        Rx_all.append(Rx)
        #print(np.shape(Rx_all))
    Rx_all= np.array(Rx_all)    

    return f_bins, Rx_all

fs = 48000
signal = wavaudioread("Module3\Recordings\d1source_60degreesv2.wav",fs)
th_range = np.linspace(-np.pi/2,np.pi/2, 1000)
nperseg = 300
N_Bins = 150

Mics = 6
d = 0.1
v=340

f_bins, Rx_all = narrowband_Rx(signal,nperseg,N_Bins)

Pytotal = np.zeros(len(th_range))

#Power for matched beamformer
for i in range(1,N_Bins):
    Py = matchedbeamformer(Rx_all[i,:,:],th_range,Mics,d,v,f_bins[i])
    Pytotal = Pytotal + Py
PyAvg_mbf = Pytotal / N_Bins

Pytotal = np.zeros(len(th_range))

#Power for mvdr
for i in range(1,N_Bins):
    Py = mvdr(Rx_all[i,:,:],th_range,Mics,d,v,f_bins[i])
    Pytotal = Pytotal + Py
PyAvg_mvdr = Pytotal / N_Bins

print(len(f_bins))
print(f_bins[1])
#plt.rcParams['figure.figsize'] = [7 , 5]
#plt.rcParams['figure.dpi'] = 150
plt.subplot(211)
plt.plot(th_range*180/np.pi,abs(PyAvg_mbf))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial spectrum, Matched Beamformer") 
plt.vlines(60,ymin=min(PyAvg_mbf),ymax=max(PyAvg_mbf),linestyles="dashed",colors="r")


plt.subplot(212)
plt.plot(th_range*180/np.pi,abs(PyAvg_mvdr)/max(abs(PyAvg_mvdr)))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial spectrum, MVDR") 
plt.vlines(60,ymin=0,ymax=1,linestyles="dashed",colors="r")

plt.tight_layout()
plt.show()