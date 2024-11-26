import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from wavaudioread import wavaudioread
from recording_tool import recording_tool
from IPython.display import Audio
from Functions import mvdr, matchedbeamformer
fs = 48000

signal = wavaudioread("Module3\Recordings\d1source_0degrees.wav",fs)


th_range = np.linspace(-np.pi/2,np.pi/2, 10000)

nperseg = 250
Sx_all = []
win = ('gaussian', 1e-2 * fs) # Gaussian with 0.01 s standard dev.
SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap=0,scale_to='magnitude', phase_shift=None)
f_bins = SFT.f

for i in range(0,8):
    if max(signal[:,i]) > 100:

        Mic = signal[:,i]
        Sx = SFT.stft(Mic)
        Sx_all.append(Sx)

Pytotal = np.zeros(len(th_range))


Sx_all = np.array(Sx_all)
for j in range(1,30):
    f0=f_bins[j]
    X = Sx_all[:, j, :]
    Rx = np.cov(X)
    Py = matchedbeamformer(Rx,th_range,6,0.1,340,f0)
   

    Pytotal = Pytotal + Py

Pytotal = Pytotal/ 2

#plt.rcParams['figure.figsize'] = [7 , 5]
#plt.rcParams['figure.dpi'] = 150
plt.subplot(211)
plt.plot(th_range*180/np.pi,abs(Pytotal))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial spectrum, Matched beamformer") 
plt.vlines(60,ymin=0,ymax=max(Pytotal),linestyles="dashed",colors="r")

Pytotal = np.zeros(len(th_range))


for j in range(1,30):
    f0=f_bins[j]
    X = Sx_all[:, j, :]
    Rx = np.cov(X)
    Py = mvdr(Rx,th_range,6,0.1,340,f0)
   

    Pytotal = Pytotal + Py

Pytotal = Pytotal/ 2

plt.subplot(212)
plt.plot(th_range*180/np.pi,abs(Pytotal))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial spectrum, MVDR") 
plt.vlines(60,ymin=0,ymax=max(Pytotal),linestyles="dashed",colors="r")

plt.tight_layout()
plt.show()     
