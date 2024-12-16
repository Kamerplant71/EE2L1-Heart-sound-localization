import numpy as np
import matplotlib.pyplot as plt
from wavaudioread import wavaudioread

Fs = 48000
signal = wavaudioread("module2/recording_S2_dual_channel.wav",Fs)


plt.plot(signal)
plt.show()