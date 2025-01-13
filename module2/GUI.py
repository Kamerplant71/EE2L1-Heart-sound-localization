import numpy as np
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft, ifft
import tkinter as tk
from Functionsmodule1 import peaks_baby, Impulse, zero_pad_arrays, h_combine, input_signal, bpm_more_signals, ch3, SEE_plot, Data_plot, plot_spectrogram

def reset_to_initial_state():
    # Reset the interface to the initial state with Signal 1 and Signal 2 buttons.
    global info_label, signal_own # Use global so we can access it for destruction later

    # Clear the window by destroying all widgets
    for widget in master.winfo_children():
        widget.destroy()

    # Recreate initial interface
    info_label = tk.Label(master, text="What data would you like to use?")
    info_label.grid(row=0, column=1)
    info2_label = tk.Label(master, text="Your own signal. copy relative path and change forward / to backwards")
    info2_label.grid(row=3,column=1)

    tk.Button(master, text="Signal 1", command=lambda: data_chooser(1)).grid(row=1, column=1)
    tk.Button(master, text="Signal 2", command=lambda: data_chooser(2)).grid(row=2, column=1)
    tk.Button(master,text="Confirm", command = signal_getten).grid(row=4, column=2)
    
    signal_own = tk.Entry(master)
    signal_own.grid(row=4, column=1)


def signal_getten():
    global so, Fs, x
    try:
        so = signal_own.get()  # Retrieve the custom signal path from the Entry widget
        if so:  # If a custom path is provided
            Fs, x = wavfile.read(so)  # Load the signal
            print(f"Custom signal loaded: {so}")
            print(f"Sampling rate: {Fs}, Signal length: {len(x)}")
            show_plotting_options()  # Proceed to the plotting options interface
        else:
            print("No custom signal path provided.")
    except FileNotFoundError:
        print(f"File not found: {so}")
    except ValueError as e:
        print(f"Error reading signal: {e}")

 
def distance_getten():
    global d
    # Retrieve the value from the Entry widget
    try:
        distance_value = distance.get()  # Use the global `distance` reference
        d = int(distance_value)  # Convert the input to an integer
        print("Minimal distance between peaks:", d)
    except ValueError:
        print("Please enter a valid number.")

    peaks_baby(Fs,x,d)

def peaks():
    global distance  # Declare `distance` as global to access it in `distance_getten`
    
    # Clear all widgets from the master window
    for widget in master.winfo_children():
        widget.destroy()
    
    # Add new label and entry for minimal distance
    tk.Label(master, text="Minimal distance between peaks").grid(row=0, column=0, padx=10, pady=10)
    
    # Create and store the Entry widget reference
    distance = tk.Entry(master)
    distance.grid(row=0, column=1, padx=10, pady=10)
    
    # Add a confirm button to retrieve the entry value
    tk.Button(master, text="Confirm", command=distance_getten).grid(row=1, column=0, columnspan=2, pady=10)

    # Add a "Go Back" button
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=3)


def data_chooser(value):
    # Handle signal selection and display plotting options.
    global Fs, x, info_label, info2_label

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

    



def show_plotting_options():
     # Clear all widgets from the master window
    for widget in master.winfo_children():
        widget.destroy()
    

    # Display options for plotting after a signal is loaded.
    tk.Label(master, text="What plot would you like?").grid(row=0, column=1)
    tk.Button(master, text="Data", command=lambda: handle_button_action("Data")).grid(row=1, column=1)
    tk.Button(master, text="Spectrogram", command=lambda: handle_button_action("Spectrogram")).grid(row=2, column=1)
    tk.Button(master, text="Shannon Energy", command=lambda: handle_button_action("Shannon energy")).grid(row=3, column=1)

    # Add "Go Back" button
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=2)

    # Peaks option
    tk.Button(master, text="Peaks", command=peaks).grid(row=5, column=1)


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
info2_label = None
Fs = None
x = None
d = None
so= None

# Initial interface setup
reset_to_initial_state()

# Run the Tkinter main loop
master.mainloop()