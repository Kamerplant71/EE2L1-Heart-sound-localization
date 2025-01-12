import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft,ifft
import tkinter as tk
from tkinter import ttk
from Functionsmodule1 import peaks_baby, Impulse, zero_pad_arrays, h_combine, input_signal, bpm_more_signals, ch3, SEE_plot



def data_chooser(value):
    if value == 1:
        Fs, x = wavfile.read("Module1\hrecording_heart_ownDevice.wav")
        print("You chose signal 1")
        

    elif value == 2:
        print("You chose signal 2")
        Fs,x = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")
        


# def FS():  # Handles retrieving entry values
#     FS = e1.get()
#     #distance = e2.get()
#     print(f"Sample Frequency: {FS}")
#     #print(f"Minimal Distance: {distance}")


def distance():
    distance = e2.get()
    print(f"Minimal Distance: {distance}")


def handle_button_action(action):  # Handles button actions
    #print(action)
    if action == "Spectrogram":
        print("Heartbeat action triggered")


        
    elif action == "Shannon energy":
        print("Shannon energy action triggered")
        SEE_plot(Fs,x)
        
    # elif action == "SEE":
    #     print("SEE action triggered")
        
master = tk.Tk()
master.title("Heartbeat")
master.geometry("700x250")

# Create buttons for the signals
signal_1 = tk.Button(master, text="signal 1", command=lambda: data_chooser(1)).grid(row=1, column=1)
signal_2 = tk.Button(master, text="signal 2", command=lambda: data_chooser(2)).grid(row=2, column=1)


# Create labels and entry
# e1 = tk.Entry(master)
# e1.grid(row=1, column=1)
e2 = tk.Entry(master).grid(row=11, column=1)

# Information
info1 = tk.Label(master, text="what data would you like to use?").grid(row=0, column=1)
info2 = tk.Label(master, text="What plot would you like?").grid(row=5, column=1)
info3 = tk.Label(master, text="Minimal distance between the peaks to be?").grid(row=10, column=1)

# Create buttons
# confirm_button_FS = tk.Button(master, text="Confirm FS", command=FS).grid(row=1, column=2)
confirm_button_distance = tk.Button(master, text="Confirm Distance", command=distance).grid(row=11, column=2)
heartsignal_button = tk.Button(master, text="Spectrogram", command=lambda: handle_button_action("Heartbeat")).grid(row=6, column=1)
se_button = tk.Button(master, text="Shannon energy", command=lambda: handle_button_action("Shannon energy")).grid(row=7, column=1)
# SEE_button = tk.Button(master, text="SEE", command=lambda: handle_button_action("SEE")).grid(row=8, column=1)

master.mainloop()