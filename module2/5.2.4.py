import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, TransferFunction, lsim, tf2zpk
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
Fs = 4000
Ts = 1/Fs


def Transfer(duration,frequency,relative_amplitude,delay):
    a = 1/duration
    omega = 2*np.pi*frequency
    system = TransferFunction([0,0,omega*relative_amplitude],[1,2*a,a**2+omega**2])
    t, h_almost = signal.impulse(system) # impulse response
    print(t.shape)
    h = np.concatenate((np.zeros(int(delay*Fs)),h_almost))
    t = np.linspace(0,len(t)+delay*Fs,len(h))
    print(f"tconcatenated:{t.shape}")
    #print(t.shape)
    #w, H = signal.freqresp(system) # Frequency response
    H = fft(h)
    w = np.linspace(0,2*np.pi*frequency,len(H))
    return t,h,w,H

def zero_pad_arrays(t1,t2,t3,t4):
    mlen = max(map(len, [t1, t2, t3,t4]))
    t1_new = np.concatenate((t1,np.zeros(mlen-len(t1))))
    t2_new = np.concatenate((t2,np.zeros(mlen-len(t2))))
    t3_new = np.concatenate((t3,np.zeros(mlen-len(t3))))
    t4_new = np.concatenate((t4,np.zeros(mlen-len(t4))))
    return t1_new, t2_new, t3_new, t4_new

t_M,h_M,w_M,H_M = Transfer(0.02,50,1,0.01)
t_T,h_T,w_T,H_T = Transfer(0.02,150,0.5,0.04)
t_A,h_A,w_A,H_A = Transfer(0.02,50,0.5,0.3)
t_P,h_P,w_P,H_P = Transfer(0.02,30,0.4,0.33)
print(np.shape(t_M))
t_M,t_T,t_A,t_P = zero_pad_arrays(t_M,t_T,t_A,t_P)
h_M,h_T,h_A,h_P = zero_pad_arrays(h_M,h_T,h_A,h_P)
w_M,w_T,w_A,w_P = zero_pad_arrays(w_M,w_T,w_A,w_P)
H_M,H_T,H_A,H_P = zero_pad_arrays(H_M,H_T,H_A,H_P)

t = t_M+t_T+t_A+t_P
h = h_M+h_T+h_A+h_P
w = w_M+w_T+w_A+w_P
H = H_M+H_T+H_A+H_P

plt.subplot(211)
plt.plot(t,h)
plt.xlabel("Time(s)")
plt.ylabel("Magnitude")
plt.title("Impulse response")
#plt.xlim(0 ,3)
plt.tight_layout()

plt.subplot(212)
plt.plot(w,H)
plt.xlabel("Frequency(Rad/s)")
plt.ylabel("Magnitude")
plt.title("Transfer function")
#plt.xlim(0,20)
plt.show()
plt.tight_layout()