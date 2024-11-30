import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, TransferFunction, lsim, tf2zpk
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
Fs = 4000
Velocity = 60*100 #in cms
def Impulse(duration,frequency,relative_amplitude,delay):
    a = 1/duration
    omega = 2*np.pi*frequency
    system = TransferFunction([0,0,omega*relative_amplitude],[1,2*a,a**2+omega**2])
    t, h_almost = signal.impulse(system) # impulse response
    h = np.concatenate((np.zeros(int(delay*Fs)),h_almost))
    return h

h_M = Impulse(0.02,50,1,0.01)
h_T = Impulse(0.02,150,0.5,0.04)
h_A = Impulse(0.02,50,0.5,0.3)
h_P = Impulse(0.02,30,0.4,0.33)

def zero_pad_arrays(t1,t2,t3,t4):
    mlen = max(map(len, [t1, t2, t3,t4]))
    t1_new = np.concatenate((t1,np.zeros(mlen-len(t1))))
    t2_new = np.concatenate((t2,np.zeros(mlen-len(t2))))
    t3_new = np.concatenate((t3,np.zeros(mlen-len(t3))))
    t4_new = np.concatenate((t4,np.zeros(mlen-len(t4))))
    return t1_new, t2_new, t3_new, t4_new



h_M,h_T,h_A,h_P = zero_pad_arrays(h_M,h_T,h_A,h_P)
# Valve positions (x, y, z) in cm
Valve_locations = np.array([
    [6.37, 10.65, 6.00],  # Valve M
    [0.94, 9.57, 5.50],   # Valve T
    [5.50, 11.00, 3.60],  # Valve A
    [3.90, 11.50, 4.50]   # Valve P
])

# Microphone positions (x, y, z) in cm
Mics = np.array([
    [2.5, 5.0, 0.0],   # Mic 0
    [2.5, 10.0, 0.0],  # Mic 1
    [2.5, 15.0, 0.0],  # Mic 2
    [7.5, 5.0, 0.0],   # Mic 3
    [7.5, 10.0, 0.0],  # Mic 4
    [7.5, 15.0, 0.0]   # Mic 5
])
distances = np.linalg.norm(Valve_locations[:, np.newaxis, :] - Mics[np.newaxis, :, :], axis=2)
Gain = 1/distances
delays = distances/Velocity

valve_signals = [h_M,h_T,h_A,h_P]

X = np.zeros([6,len(h_A)])
for m in range(6):  # For each microphone
    for v in range(4):  # For each valve
        Delay_apply_sig = np.concatenate((np.zeros(int(delays[v,m]*Fs)),valve_signals[v]))




print (X)
