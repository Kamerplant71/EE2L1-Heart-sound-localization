import numpy as np
import matplotlib.pyplot as plt
from wavaudioread import wavaudioread

from Functions import music, narrowband_Rx


fs = 48000
signal = wavaudioread("Module3\Recordings\d1source_60degreesv2.wav",fs)
th_range = np.linspace(-np.pi/2,np.pi/2, 1000)
nperseg = 350
N_Bins = 60


Q = 1
Mics = 6
d = 0.1
v=340

f_bins, Rx_all = narrowband_Rx(signal,nperseg)
print(len(f_bins))
Pytotal = np.zeros(len(th_range))

#Power for matched beamformer
for i in range(1,N_Bins):
    Py = music(Rx_all[i,:,:],Q,Mics,th_range,d,v,f_bins[i])
    Pytotal = Pytotal + Py
PyAvg_music = Pytotal / N_Bins

print(f_bins[1])
plt.plot(th_range*180/np.pi,(abs(PyAvg_music)))
plt.xlabel("Angle [deg]")
plt.ylabel("Power ")
plt.title("Spatial spectrum, MUSIC")
plt.vlines(60,ymin=min(PyAvg_music),ymax=max(PyAvg_music),linestyles="dashed",colors="r")

plt.show()
