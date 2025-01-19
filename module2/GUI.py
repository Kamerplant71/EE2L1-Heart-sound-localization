import numpy as np
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft, ifft
import tkinter as tk
from Functionsmodule1 import peaks_baby, Impulse, zero_pad_arrays, h_combine, input_signal, bpm_more_signals, ch3, SEE_plot, Data_plot, plot_spectrogram, cutting

def reset_to_initial_state():
    # Reset the interface to the initial state with Signal 1 and Signal 2 buttons.
    global info_label, signal_own # Use global so we can access it for destruction later

    # Clear the window by destroying all widgets
    for widget in master.winfo_children():
        widget.destroy()

    # Recreate initial interface
    info_label = tk.Label(master, text="What example data would you like to use?")
    info_label.grid(row=0, column=1)
    info2_label = tk.Label(master, text="Your own signal. Copy relative path and change \ to /")
    info2_label.grid(row=3,column=1)

    tk.Button(master, text="Signal 1", command=lambda: data_chooser(1)).grid(row=1, column=1, padx=10, pady=2)
    tk.Button(master, text="Signal 2", command=lambda: data_chooser(2)).grid(row=2, column=1, padx=10, pady=2)
    tk.Button(master,text="Confirm", command = signal_getten).grid(row=4, column=2,padx=10, pady=10, sticky="w")
    
    signal_own = tk.Entry(master, width=40)
    signal_own.grid(row=4, column=1)


def signal_getten():
    global so, Fs, x
    so = signal_own.get()  # Retrieve the custom signal path from the Entry widget
    if so:  # If a custom path is provided
        Fs, x = wavfile.read(so)  # Load the signal
        print(f"Custom signal loaded: {so}")
        print(f"Sampling rate: {Fs}, Signal length: {len(x)}")
        show_plotting_options()  # Proceed to the plotting options interface
    else:
        print("No custom signal path provided.")
  
    
 
def distance_getten():
    global d, h
    # Retrieve the value from the Entry widget
    try:
        distance_value = distance.get()  # Use the global `distance` reference
        d = int(distance_value)  # Convert the input to an integer
        print("Minimal distance between peaks:", d)
    except ValueError:
        print("Please enter a valid number.")

    try:
        height_value = height.get()  # Get the height value from the Entry widget
        h = float(height_value)  # Convert to integer for height
    except ValueError:
        print("Please enter a valid number for height.")
        return  # If the height is invalid, stop the function
    
    peaks_baby(Fs,x,d, h )

def peaks():
    global distance, height  # Declare `distance` as global to access it in `distance_getten`
    
    # Clear all widgets from the master window
    for widget in master.winfo_children():
        widget.destroy()
    
    # Add new label and entry for minimal distance
    tk.Label(master, text="Set two variables").grid(row=0, column=0, padx=10, pady=2)
    tk.Label(master, text="Minimal distance between peaks").grid(row=1, column=0, padx=10, pady=2)
    tk.Label(master, text="Threshold for the peaks").grid(row=2, column=0, padx=10, pady=2)

    # Create and store the Entry widget reference
    distance = tk.Entry(master)
    distance.grid(row=1, column=1, padx=10, pady=2)

    height = tk.Entry(master)
    height.grid(row=2, column=1, padx=10, pady=2)
    
    # Add a confirm button to retrieve the entry value
    tk.Button(master, text="Confirm", command=distance_getten).grid(row=3, column=0, columnspan=2, pady=10)

    # Add a "Go Back" button
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=3)


def slicing():
    global first_entry, second_entry

    for widget in master.winfo_children():
        widget.destroy()

    tk.Label(master, text="Shannon Energy Envelope analysis").grid(row=0, column=0,columnspan=4,pady=2)
    tk.Label(master, text="Cut sections in time intervals").grid(row=2, column=0, columnspan=3, pady=2)
    tk.Label(master, text="Start Time").grid(row=4, column=0, padx=5, pady=2, sticky="e")
    tk.Label(master, text="End Time").grid(row=4, column=1, padx=5, pady=2, sticky="w")

    
    tk.Button(master, text="Plot uncutted SEE", command=lambda: handle_button_action("SEE")).grid(row=1, column=0, columnspan=4, pady=2)
    tk.Button(master, text="Add index", command=Add_index).grid(row=3, column=0, columnspan=2, pady=2)
    tk.Button(master, text="Confirm", command=cutting_arr).grid(row=row_counter + 5, column=0, padx=10, pady=2, sticky="e")
    tk.Button(master, text="Plot cutted signal", command=plot_new_xnorm).grid(row=row_counter + 5, column=1, padx=10, pady=2, sticky="w")
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=2)

    first_entry = tk.Entry(master)
    first_entry.grid(row=5, column=0, padx=5, pady=2)
    entry_list.append(first_entry)
    second_entry = tk.Entry(master)
    second_entry.grid(row=5, column=1, padx=5, pady=2)
    entry_list.append(second_entry)



def Add_index():
    global row_counter  # Keep track of the current row number
    for i in range(1):  # Add two new entries
        first_entry = tk.Entry(master)
        first_entry.grid(row=row_counter, column=0, padx=5, pady=2)
        entry_list.append(first_entry)  # Store the reference to the entry in the list

        second_entry = tk.Entry(master)
        second_entry.grid(row=row_counter, column=1, padx=5, pady=2)
        entry_list.append(second_entry)  # Store the reference to the entry in the list

        row_counter += 1  # Increment row counter



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
        show_plotting_options()
        
    elif value == 2:
        Fs, x = wavfile.read("Module1\heart_single_channel_physionet_49829_TV.wav")
        show_plotting_options()

    
def cutting_arr():
    global xnorm
    arr1 = []
    for i in range(0, len(entry_list), 2):  # Iterate in pairs
        value_1 = entry_list[i].get()
        value_2 = entry_list[i + 1].get()
        value_1 = float(value_1)
        value_2 = float(value_2)
        arr1.append([value_1, value_2])
    
    finalarr = np.array(arr1)
    

    print("Array of values:", finalarr) 
     # Debugging output
    xnorm = cutting(x, Fs, finalarr)
    
    return finalarr

def plot_new_xnorm():
    
    SEE_plot(Fs,xnorm)




def show_plotting_options():
     # Clear all widgets from the master window
    for widget in master.winfo_children():
        widget.destroy()
    

    # Display options for plotting after a signal is loaded.
    tk.Label(master, text="Plotting options").grid(row=0, column=1, padx=5, pady=2)
    tk.Label(master, text="Edit signal").grid(row=4, column=1, padx=5, pady=2)


    
    tk.Button(master, text="The Signal", command=lambda: handle_button_action("Data")).grid(row=1, column=1, padx=5, pady=2)
    tk.Button(master, text="Spectrogram", command=lambda: handle_button_action("Spectrogram")).grid(row=2, column=1,padx=5, pady=2)
    tk.Button(master, text="Shannon Energy Envelope", command=lambda: handle_button_action("SEE")).grid(row=3, column=1,padx=5, pady=2)
    tk.Button(master, text="Slicing", command= slicing).grid(row=5, column=1,padx=5, pady=2)
    


    # Add "Go Back" button
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=2)

    # Peaks option
    tk.Button(master, text="Detect peaks", command=peaks).grid(row=6, column=1)


def handle_button_action(action):
    # Handle plot selection actions.
    global Fs, x
    if Fs is None or x is None:
        print("Please select a signal first.")
        return

    if action == "Spectrogram":
        plot_spectrogram(Fs, x)
    elif action == "Data":
        Data_plot(Fs, x)
    elif action == "SEE":
        SEE_plot(Fs, x )


# Main Tkinter window
master = tk.Tk()
master.title("Heartbeat")
master.geometry("700x500")

# Global variable for the info label
info_label = None
info2_label = None
Fs = None
x = None
d = None
h = None
so= None
xnorm= None
row_counter = 7
entry_list = []

# Initial interface setup
reset_to_initial_state()

# Run the Tkinter main loop
master.mainloop()