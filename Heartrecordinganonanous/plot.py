import numpy as np
import matplotlib.pyplot as plt
from wavaudioread import wavaudioread
fs = 48000
signal = wavaudioread("Heartrecordinganonanous\speedofsoundv2.wav",fs)
#signal= signal[:,3]
print(np.shape(signal))
plt.plot(signal)
plt.show()
