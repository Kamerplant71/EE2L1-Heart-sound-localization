import numpy as np
import matplotlib.pyplot as plt
from wavaudioread import wavaudioread

from Functions import music, narrowband_Rx2


fs = 48000
signal1 = wavaudioread("Module3\Recordings\d1source_0degrees.wav",fs)
signal2 = wavaudioread("Module3\Recordings\d1source_20degrees.wav",fs)
th_range = np.linspace(-np.pi/2,np.pi/2, 5000)
nperseg = 512
print(np.shape(signal1))
print(np.shape(signal2))

signal = signal1[:623040,] + signal2

Q = 2
Mics = 6
d = 0.1
v=340

f_bins, Rx_all, Xall = narrowband_Rx2(signal,nperseg)
print(len(f_bins))
Pytotal = np.zeros(len(th_range))
N_Bins = len(f_bins)
#Power for matched beamformer
for i in range(1,N_Bins):
    Py = music(Rx_all[i,:,:],Q,Mics,th_range,d,v,f_bins[i])
    Pytotal = Pytotal + Py

PyAvg_music = Pytotal #/ N_Bins

print(f_bins[1])
plt.plot(th_range*180/np.pi,(abs(PyAvg_music)))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial spectrum, MUSIC")
plt.vlines(60,ymin=min(PyAvg_music),ymax=max(PyAvg_music),linestyles="dashed",colors="r")

plt.show()
