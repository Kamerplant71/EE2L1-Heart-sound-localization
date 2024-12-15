import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, TransferFunction, lsim, tf2zpk
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft

Fs = 2000
Velocity = 60*100 #in cms

def Impulse(duration,frequency,relative_amplitude,delay):
    a = 1/duration
    omega = 2*np.pi*frequency
    system = TransferFunction([0,0,omega*relative_amplitude],[1,2*a,a**2+omega**2])
    t, h_almost = signal.impulse(system) # impulse response
    h = np.concatenate((np.zeros(int(delay*Fs)),h_almost))
    return h

def peaks_baby(Fs,x):
    period = 1/Fs
    xn= x/max(abs(x))
    xi = xn**2
    xi = xi + 1e-10
    E=-(xi *np.log10(xi))

    b, a = butter(2, [15/(Fs/2)], btype='low')
    y = signal.lfilter(b,a,E)# SEE envelope
    y = y.real
    Y= fft(y)
    t= np.linspace(0,period*len(y),len(y))
    f= np.linspace(0,Fs, len(Y))

    pieces = np.array_split(y, 8)
    max_values = [np.max(piece) for piece in pieces]
    mean = np.mean(max_values)
    
    for i in range(len(max_values)):
        if max_values[i]>1.5*mean:
            pieces[i] = np.zeros(len(pieces[1]))
    
    y_reconstructed = np.concatenate((pieces))
    ynorm = y_reconstructed/max(y_reconstructed)
    peaks, loveisintheair= signal.find_peaks(ynorm,0.4,distance = Fs/5)

    # plt.plot(t,ynorm)
    # plt.xlim(7,10)
    # plt.show()
    diffpeaksS1=[]
    diffpeaksS2=[]
    diffpeaks=[]
    peaksbaby =[]
    for i in range(len(peaks)-1):
        # print(i)
        diffpeaks.append(peaks[i+1] - peaks[i])

    for i in range(len(diffpeaks)-1):
        if diffpeaks[i]>diffpeaks[i+1]:
            peaksbaby.append(("Peak at", np.round(peaks[i]/Fs,2), "seconds is S2"))
        else: peaksbaby.append(("Peak at", np.round(peaks[i]/Fs,2), "seconds is S1"))

    endpeaksbaby=[]
    if len(peaks) >= 3:  
        diffpeaks[-1] = peaks[-2] - peaks[-3] 
        diffpeaks[-2] = peaks[-1] - peaks[-2]  

    if diffpeaks[-1] > diffpeaks[-2]:
        endpeaksbaby.append(("Peak at", np.round(peaks[-2]/Fs,2), "seconds is S2"))
        endpeaksbaby.append(("Peak at", np.round(peaks[-1]/Fs,2), "seconds is S1"))
    else:
        endpeaksbaby.append(("Peak at", np.round(peaks[-2]/Fs,2), "seconds is S1"))
        endpeaksbaby.append(("Peak at", np.round(peaks[-1]/Fs,2), "seconds is S2"))


    peaksbaby= np.concatenate((peaksbaby, endpeaksbaby))

    # print(peaksbaby)
    return peaksbaby
   


def zero_pad_arrays(t1,t2,t3,t4):
    mlen = max(map(len, [t1, t2, t3,t4]))
    t1_new = np.concatenate((t1,np.zeros(mlen-len(t1))))
    t2_new = np.concatenate((t2,np.zeros(mlen-len(t2))))
    t3_new = np.concatenate((t3,np.zeros(mlen-len(t3))))
    t4_new = np.concatenate((t4,np.zeros(mlen-len(t4))))
    return t1_new, t2_new, t3_new, t4_new

def input_signal(Ns,list_of_impulse_responses):
    s = [None]*len(list_of_impulse_responses)
    x = [None]*len(list_of_impulse_responses)
    for i in range(len(list_of_impulse_responses)):

        s[i]=np.concatenate((np.random.randn(Ns),np.zeros(len(list_of_impulse_responses[i]-Ns))))
        x[i]=np.convolve(s[i],list_of_impulse_responses[i],mode = "full")
    return x

# def bpm_more_signals (Matrix_of_signals,BPM):
#     BPS = BPM/60
#     X = np.zeros([6,len(np.tile(np.concatenate((Matrix_of_signals[1],np.zeros(int(BPS*Fs)))),25))])
#     for i in range(len(Matrix_of_signals)):
#         X[i].append(np.tile(np.concatenate((Matrix_of_signals[i],np.zeros(int(BPS*Fs))))),25)
#     return X

def bpm_more_signals(Matrix_of_signals, BPM):
    BPS = BPM / 60  # Convert BPM to BPS
     # Initialize a matrix with zeros
    X = np.zeros((6, (len(Matrix_of_signals[0]) + int(BPS * Fs))*10))
    
    for i in range(len(Matrix_of_signals)):
        # Concatenate and tile signals to replace zeros
        signal_extension = np.tile(np.concatenate((Matrix_of_signals[i], np.zeros(int(BPS * Fs)))), 10)
        # Update row i in matrix X
        X[i] = signal_extension  # Ensure dimensions match

    return X
        
h_M = Impulse(0.02,45,1,0.01)
h_T = Impulse(0.02,40,0.8,0.04)
h_A = Impulse(0.02,40,0.7,0.3)
h_P = Impulse(0.02,30,0.7,0.33)

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

valve_impulses = [h_M,h_T,h_A,h_P]
valve_signals = input_signal(10,valve_impulses)
X = np.zeros([6,len(valve_signals[1])])

for m in range(6):  # For each microphone
    for v in range(4):  # For each valve
        Delay_apply_sig = np.concatenate((np.zeros(int(delays[v,m])*Fs),valve_signals[v]))
        print(delays[v,m])
        X[m] += Gain[v,m] * Delay_apply_sig
        #print(f"Mic {m}, Valve {v}, Signal: {X[m]}")
        #print(np.all( X[m]== 0))
 
# Slice the array to exclude trailing zeros
last_nonzero_index = np.max(np.nonzero(X[1]))
trimmed_X = X[1][:last_nonzero_index + 1]
X_new = np.zeros([6,len(trimmed_X)])
for m in range(6):
    X_new[m, :] = X[m][:last_nonzero_index + 1]

X_BPM = bpm_more_signals(X_new,60)

# for m in range(len(X_new)):
#     t = np.linspace (0,len(X_new[m])/Fs,len(X_new[m]))
#     plt.subplot(3,2,int(m+1))
#     plt.plot(t,X_new[m])
#     plt.xlabel("Time(s)")
#     plt.ylabel("Magnitude")
#     plt.title("signal")
#     plt.xlim(0,1)
#     jaja = peaks_baby(Fs,X_new[m])

for m in range(len(X_BPM)):
    t = np.linspace (0,len(X_BPM[m])/Fs,len(X_BPM[m]))
    plt.subplot(3,2,int(m+1))
    plt.plot(t,X_BPM[m])
    plt.xlabel("Time(s)")
    plt.ylabel("Magnitude")
    plt.title("signal")
    plt.xlim(0,3)
    jaja = peaks_baby(Fs,X_BPM[m])

plt.tight_layout()
plt.show()
#print(distances)
#print(delays)
#print(np.all(Delay_apply_sig == 0))  # Returns True if all elements are zero
#print(distances)