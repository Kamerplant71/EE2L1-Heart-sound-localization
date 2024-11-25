import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from wavaudioread import wavaudioread
from recording_tool import recording_tool
from IPython.display import Audio

fs = 48000

signal = wavaudioread("Module3\Recordings\d1source_0degrees.wav",fs)

def narrowband_Rx(signal,nperseg):
    
    
    win = ('gaussian', 1e-2 * fs) # Gaussian with 0.01 s standard dev.
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap=0, scale_to='magnitude', phase_shift=None)
    Sx_all = []
    for mic in range(0,7):

        Sx = SFT.stft(signal[:][mic])
        f_bins = SFT.f
        Sx_all.append(Sx)

    return f_bins

f_bins= narrowband_Rx(signal,10)
print(np.shape(f_bins))
print(f_bins)