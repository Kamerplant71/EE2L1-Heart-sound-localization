import numpy as np
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft, ifft
import tkinter as tk
from Functionsmodule1 import peaks_baby, Impulse, zero_pad_arrays, h_combine, input_signal, bpm_more_signals, ch3, SEE_plot, Data_plot, plot_spectrogram

Fs = None
x = None


def reset_to_initial_state():
    # Reset the interface to the initial state with Signal 1 and Signal 2 buttons.
    global info_label  # Use global so we can access it for destruction later

    # Clear the window by destroying all widgets
    for widget in master.winfo_children():
        widget.destroy()

    # Recreate initial interface
    info_label = tk.Label(master, text="What data would you like to use?")
    info_label.grid(row=0, column=1)

    tk.Button(master, text="Signal 1", command=lambda: data_chooser(1)).grid(row=1, column=1)
    tk.Button(master, text="Signal 2", command=lambda: data_chooser(2)).grid(row=2, column=1)


def distance_getten(distance):
    
    # Retrieve the value from the Entry widget
    distance_value = distance.get()
    d = int(distance_value)

    # Print the value
    print("Minimal distance between peaks:", distance_value)

    peaks_baby(Fs,x,d)


def peaks_baby():
    # Clear all widgets from the master window
    for widget in master.winfo_children():
        widget.destroy()
    
    # Add new label and entry for minimal distance
    tk.Label(master, text="Minimal distance between peaks").grid(row=0, column=0, padx=10, pady=10)
    distance = tk.Entry(master)
    distance.grid(row=0, column=1, padx=10, pady=10)
    
    # Add a confirm button to retrieve the entry value
    tk.Button(master, text="Confirm", command=lambda: distance_getten(distance)).grid(row=1, column=0, columnspan=2, pady=10)

    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=3)


def data_chooser(value):
    # Handle signal selection and display plotting options.
    global Fs, x, info_label

    # Destroy all widgets (clear current interface)
    for widget in master.winfo_children():
        widget.destroy()

    # Destroy the info label if it exists
    if info_label:
        info_label.destroy()

    # Load the selected signal
    if value == 1:
        Fs, x = wavfile.read("Module1\hrecording_heart_ownDevice.wav")
        
    elif value == 2:
        Fs, x = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")


    # Add new options for plotting
    tk.Label(master, text="What plot would you like?").grid(row=0, column=1)

    tk.Button(master, text="Data", command=lambda: handle_button_action("Data")).grid(row=1, column=1)
    tk.Button(master, text="Spectrogram", command=lambda: handle_button_action("Spectrogram")).grid(row=2, column=1)
    tk.Button(master, text="Shannon Energy", command=lambda: handle_button_action("Shannon energy")).grid(row=3, column=1)

    # Add "Go Back" button
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=2)

    # Peaksbaby
    tk.Button(master, text="Peaks", command=peaks_baby).grid(row=5 ,column=1)


def handle_button_action(action):
    # Handle plot selection actions.
    global Fs, x
    if Fs is None or x is None:
        print("Please select a signal first.")
        return

    if action == "Spectrogram":
        plot_spectrogram(Fs, x)
    elif action == "Shannon energy":
        SEE_plot(Fs, x)
    elif action == "Data":
        Data_plot(Fs, x)


# Main Tkinter window
master = tk.Tk()
master.title("Heartbeat")
master.geometry("700x250")

# Global variable for the info label
info_label = None

# Initial interface setup
reset_to_initial_state()

# Run the Tkinter main loop
master.mainloop()