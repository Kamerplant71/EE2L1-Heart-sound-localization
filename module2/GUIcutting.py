import numpy as np
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft, ifft
import tkinter as tk
from Functionsmodule1 import peaks_baby, Impulse, zero_pad_arrays, h_combine, input_signal, bpm_more_signals, ch3, SEE_plot, Data_plot, plot_spectrogram


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
    tk.Label(master, text="plot the SEE").grid(row=0, column=1)
    tk.Label(master, text="What section do you want to cut? (t)").grid(row=4,column=1)


    tk.Button(master, text="SEE", command=lambda: handle_button_action("SEE")).grid(row=3, column=1)
    tk.Button(master, text="Add index", command=Add_index).grid(row=row_counter+10, column=1)
    
    first_entry = tk.Entry(master).grid(row=5, column=1)
    second_entry = tk.Entry(master).grid(row=5, column=2)
    
    

    # Add "Go Back" button
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=2)


def Add_index():
    global row_counter  # Keep track of the current row number
    for i in range(1):  # Add two new entries
        first_entry = tk.Entry(master).grid(row=row_counter, column=1)
        entry_list.append(first_entry)  # Store the reference to the entry in the list

        second_entry = tk.Entry(master).grid(row=row_counter, column=2)
        entry_list.append(second_entry)  # Store the reference to the entry in the list

        row_counter += 1  # Increment row counter


def reset_to_initial_state():
    # Clear the GUI and reset to the initial state
    for entry in entry_list:
        entry.destroy()  # Remove each entry widget
    entry_list.clear()
    global row_counter
    row_counter = 6  # Reset the row counter


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


def handle_button_action(action):
    # Handle plot selection actions.
    global Fs, x
    if Fs is None or x is None:
        print("Please select a signal first.")
        return

    elif action == "SEE":
        SEE_plot(Fs, x)



# Main Tkinter window
master = tk.Tk()
master.title("Cutting")
master.geometry("700x250")

# Global variable for the info label
info_label = None
row_counter = 6
entry_list = []


# Initial interface setup
reset_to_initial_state()

# Run the Tkinter main loop
master.mainloop()

