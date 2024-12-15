import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, TransferFunction, lsim, tf2zpk
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft,ifft
from IPython.display import Audio
from playsound import playsound

# Sampling frequency
Fs = 48000  # Standard audio sampling rate

def Impulse(duration, frequency, relative_amplitude, delay):
    a = 1 / duration
    omega = 2 * np.pi * frequency
    
    # Correct the transfer function coefficients
    numerator = [relative_amplitude * omega]
    denominator = [1, 2 * a, a ** 2 + omega ** 2]
    
    # Define and compute impulse response
    system = TransferFunction(numerator, denominator)
    t, h_almost = signal.impulse(system, T=np.linspace(0, 1, int(Fs)))
    
    # Add delay with padding
    h = np.concatenate((np.zeros(int(delay * Fs)), h_almost))
    return h[:int(0.5 * Fs)]  # Limit to 0.5 seconds

def bpm_more_signals(signal, BPM):
    BPS = BPM / 60  # Beats per second
    silence = np.zeros(int(Fs * (1 / BPS)))  # Silence between beats
    return np.tile(np.concatenate((signal, silence)), 10)

# Generate impulse response and BPM signals
h_M = Impulse(0.02, 45, 1, 0.01)  # Generate single impulse response
h_bpm = bpm_more_signals(h_M, 60)  # Repeat signal with BPM structure

# Create stereo data
h_dual = np.stack([h_bpm, h_bpm], axis=-1)  # Stereo: shape (N, 2)
h_dual = np.asarray(h_dual, dtype=np.float32)  # Ensure proper data type

# Write WAV file
wavfile.write("S1.wav", Fs, h_dual)

# Play the WAV file
Audio("S1.wav",autoplay=True, rate=Fs)