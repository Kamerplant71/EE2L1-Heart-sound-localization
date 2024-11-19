import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse
from IPython.display import Audio



from refsignal import refsignal            # model for the EPO4 audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool

# Define the sampling rate and load the recording.

Fs1, y = wavfile.read("chapter_2\h13918_AV.wav")
Fs2, x = wavfile.read("chapter_2\h13918_MV.wav")
Fs3, z = wavfile.read("chapter_2\h13918_TV.wav")
Fs4, a = wavfile.read("chapter_2\h14241_AV.wav")

Y= fft(y)
X= fft(x)
Z= fft(z)
A= fft(a)

fig, ax = plt.subplots(4, 2, figsize=(12,10))

# Calculate the period, and the time [s] and frequency [kHz] axis
period1 = 1/Fs1
period2 = 1/Fs2
period3 = 1/Fs3
period4 = 1/Fs4

t1 = np.linspace(0,period1*len(y),len(y))
f1 = np.linspace(0,Fs1, len(Y))

t2 = np.linspace(0,period2*len(x),len(x))
f2 = np.linspace(0,Fs2, len(X))

t3 = np.linspace(0,period3*len(z),len(z))
f3 = np.linspace(0,Fs3, len(Z))

t4 = np.linspace(0,period4*len(a),len(a))
f4 = np.linspace(0,Fs4, len(A))

plt.subplot(421)
plt.plot(t1,y)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot 1")

plt.subplot(422)
plt.plot(f1,abs(Y))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency 1")

plt.subplot(423)
plt.plot(t2,x)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot 2")

plt.subplot(424)
plt.plot(f2,abs(X))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency 2")

plt.subplot(425)
plt.plot(t3,z)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot 3")

plt.subplot(426)
plt.plot(f3,abs(Z))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency 3")

plt.subplot(427)
plt.plot(t4,a)
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("Time plot 4")

plt.subplot(428)
plt.plot(f4,abs(A))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency 4")


# Plot the first channel of the recording in the time domain.

plt.tight_layout()
plt.show()
