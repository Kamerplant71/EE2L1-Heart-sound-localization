import numpy as np
import matplotlib.pyplot as plt
from wavaudioread import wavaudioread

from Functions import music, narrowband_Rx2


fs = 48000
signal1= wavaudioread("Module3\Recordings\d1source_0degrees.wav",fs)
signal2 = wavaudioread("Module3\Recordings\d1source_20degrees.wav",fs)
th_range = np.linspace(-np.pi/2,np.pi/2, 5000)
nperseg = 100
#print(np.shape(signal1))
#print(np.shape(signal2))

signal = signal1[:623040,] + signal2

Q = 2
Mics = 6
d = 0.1
v=340

f_bins, Rx_all, Xall = narrowband_Rx2(signal,nperseg)
print(len(f_bins))
Pytotal = np.zeros(len(th_range))
N_Bins = len(f_bins)

#Averaging the power in case of a real signal
for i in range(1,len(f_bins)):
    Py = music(Rx_all[i,:,:],Q,Mics,th_range,d,v,f_bins[i])
    Pytotal = Pytotal + Py

PyAvg_music = Pytotal /len(f_bins)

plt.rcParams['figure.figsize'] = [7 , 6]
plt.rcParams['figure.dpi'] = 150

print(f_bins[1])
plt.plot(th_range*180/np.pi,(abs(PyAvg_music))/max(abs(PyAvg_music)))
plt.xlabel("Angle [deg]",fontsize=20)
plt.ylabel("Normalized Power",fontsize=20)
plt.title("Spatial spectrum, MUSIC",fontsize=20)
plt.vlines(0,ymin=0,ymax=1,linestyles="dashed",colors="r")
plt.vlines(20,ymin=0,ymax=1,linestyles="dashed",colors="r")
plt.legend(["Power", "Expected angle"],loc= "upper left",fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

for i in range(len(th_range)):
    if PyAvg_music[i] == max(PyAvg_music):
        print(th_range[i]*180/np.pi)

plt.show()
