import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Fs, data = wavfile.read()
b, a = butter(2, [10, 800], btype='band')
filtered_data = signal.filtfilt(b, a, data)
