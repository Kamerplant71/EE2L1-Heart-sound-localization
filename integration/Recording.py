import numpy as np
import matplotlib.pyplot as plt
from wavaudioread import wavaudioread

Fs = 48000
signal = wavaudioread("recordings/recording_S2_dual_channel.wav",Fs)


plt.plot(signal)
plt.show()