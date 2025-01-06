import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, TransferFunction, lsim, tf2zpk
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

Fs = 48000
period = 1/Fs
signals = wavaudioread("integration/HumanData (2)/240508_151547/TRACK03.WAV",Fs)
print(np.shape(signals))
# RMS, max_index = rms_value(signals)
# strongest_signal = signals[:, max_index]

normalized_signal = signals/max(signals)

xi = normalized_signal**2
xi = xi + 1e-10
E=-(xi *np.log10(xi))
b, a = butter(2, [15/24000], btype='low')
y = signal.lfilter(b,a,E) #SEE envolope

t= np.linspace(0,period*len(y),len(y))

# print(RMS)
plt.plot(t,y)
plt.show()


