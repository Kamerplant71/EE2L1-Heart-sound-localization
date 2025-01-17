import numpy as np
from scipy.signal import butter, spectrogram
from scipy import signal
import math

from scipy.io import wavfile
from scipy.fft import fft, ifft
import tkinter as tk
from module2.Functionsmodule1 import peaks_baby, Impulse, zero_pad_arrays, h_combine, input_signal, bpm_more_signals, ch3, SEE_plot, Data_plot, plot_spectrogram

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from Localization3D.wavaudioread import wavaudioread
from tkinter import ttk
from Localization3D.Functions import music_z, create_points, narrowband_Rx2
fs=48000

def powercalculation(Q, Mics, v, x_steps, zmin, zmax, signal, nperseg, fbin):
    
    
    mic_positions= np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]]) /100
    
    y_steps = x_steps
    xmax = 10 /100
    ymax = 20 /100
    z = np.linspace(zmin,zmax,10)/100
    f_bins, Rx_all, Xall = narrowband_Rx2(signal,nperseg)
    print(f_bins)

    N_Bins = len(f_bins)

    Pytotal = []

    for i in range(len(z)):  # z ranges from 0 to 10
        xyz = create_points(x_steps, y_steps, xmax, ymax, z[i])
        Py_1_layer = np.zeros((x_steps,y_steps))
        # Compute the MUSIC spectrum for the current z-plane
        for i in range(1,len(f_bins)):  # len(f_bins)
            if f_bins[i] == fbin:
                Py = music_z(Rx_all[i,:,:],Q,Mics,xyz,v,f_bins[i],mic_positions,x_steps,y_steps)
                #Py = mvdr_z(Rx_all[i,:,:], Mics,xyz,v,f_bins[i],mic_positions,x_steps,y_steps)
                Py_1_layer = Py_1_layer + Py

        print(np.shape(Py_1_layer))
        Pytotal.append(Py_1_layer)
    
    Pytotal = np.array(Pytotal)
    print(np.shape(Pytotal))    
    #pymax = np.max(py)
    return Pytotal, z



# Plot the result
def plotting(Pytotal, z, xmax, ymax, mic_positions):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for i in range(len(z)):
        ax = axes[i]
        im = ax.imshow(np.abs(Pytotal[i,:,:]), extent=(0, xmax * 100, 0, ymax * 100), origin='lower', vmin=0, vmax = np.max(abs(Pytotal)))
        ax.set_title(f"z = {z[i] * 100:.1f} cm")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        mic_x = mic_positions[:, 0] * 100  # Convert to cm
        mic_y = mic_positions[:, 1] * 100  # Convert to cm
        ax.scatter(mic_x, mic_y, color='red', label='Microphones')

    # Add a colorbar to the figure
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    plt.suptitle("MUSIC Spectrum at Different z-Planes", fontsize=16)
    plt.show()


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
    so = signal_own.get()  # Retrieve the custom signal path from the Entry widget
    if so:  # If a custom path is provided
        Fs, x = wavfile.read(so)  # Load the signal
        print(f"Custom signal loaded: {so}")
        print(f"Sampling rate: {Fs}, Signal length: {len(x)}")
        show_plotting_options()  # Proceed to the plotting options interface
    else:
        print("No custom signal path provided.")
  
    
 
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


def SEE():
    global first_entry, second_entry

    for widget in master.winfo_children():
        widget.destroy()

    tk.Label(master, text="plot the SEE").grid(row=0, column=1)
    tk.Label(master, text="What section do you want to cut? (t)").grid(row=4,column=1)

    
    tk.Button(master, text="SEE", command=lambda: handle_button_action("SEE")).grid(row=3, column=1)
    tk.Button(master, text="Add index", command=Add_index).grid(row=row_counter + 10, column=1)
    tk.Button(master, text="Confirm", command=cutting_arr).grid(row=row_counter + 11, column=2)
    
    first_entry = tk.Entry(master)
    first_entry.grid(row=5, column=1)
    entry_list.append(first_entry)
    second_entry = tk.Entry(master)
    second_entry.grid(row=5, column=2)
    entry_list.append(second_entry)



def Add_index():
    global row_counter  # Keep track of the current row number
    for i in range(1):  # Add two new entries
        first_entry = tk.Entry(master)
        first_entry.grid(row=row_counter, column=1)
        entry_list.append(first_entry)  # Store the reference to the entry in the list

        second_entry = tk.Entry(master)
        second_entry.grid(row=row_counter, column=2)
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
    arr1 = []
    for i in range(0, len(entry_list), 2):  # Iterate in pairs
        value_1 = entry_list[i].get()
        value_2 = entry_list[i + 1].get()
        arr1.append([value_1, value_2])
    
    print("Array of values:", arr1) 
    signal = reinfunctie(value_1, value_2)
    
    class PowerCalculationGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("Power Calculation GUI")
            self.setup_gui()

        def setup_gui(self):
            # Input frame
            input_frame = ttk.LabelFrame(self.root, text="Input Parameters")
            input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            self.inputs = {}
            params = [
                ("Q", "1"),
                ("Mics", "6"),
                ("v (velocity)", "80"),
                ("x_steps", "50"),
                ("zmin", "6"),
                ("zmax", "15"),
                ("nperseg", "200"),
                ("fbin", "2400")
            ]

            for i, (label, default) in enumerate(params):
                ttk.Label(input_frame, text=f"{label}:").grid(row=i, column=0, padx=5, pady=5, sticky="e")
                entry = ttk.Entry(input_frame)
                entry.insert(0, default)
                entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
                self.inputs[label] = entry

            # Buttons
            action_frame = ttk.Frame(self.root)
            action_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            ttk.Button(action_frame, text="Run", command=self.run_calculation).grid(row=0, column=0, padx=5, pady=5)
            ttk.Button(action_frame, text="Exit", command=self.root.quit).grid(row=0, column=1, padx=5, pady=5)

            # Output
            self.output_label = ttk.Label(self.root, text="", wraplength=400)
            self.output_label.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        def run_calculation(self):
            try:
                # Retrieve input values
                Q = int(self.inputs["Q"].get())
                Mics = int(self.inputs["Mics"].get())
                v = float(self.inputs["v (velocity)"].get())
                x_steps = int(self.inputs["x_steps"].get())
                zmin = float(self.inputs["zmin"].get())
                zmax = float(self.inputs["zmax"].get())
                nperseg = int(self.inputs["nperseg"].get())
                fbin = int(self.inputs["fbin"].get())

                # Generate dummy signal data

                # Call the powercalculation function
                xmax = 10 /100
                ymax = 20 /100
                mic_positions= np.array( [[2.5 ,5.0, 0],
                    [2.5,10,0 ],
                    [2.5,15,0 ],
                    [7.5,5,0 ],
                    [7.5,10,0],
                    [7.5,15,0]]) /100
                Pytotal,z = powercalculation(Q, Mics, v, x_steps, zmin, zmax, signal, nperseg, fbin)
                plotting(Pytotal,z, xmax, ymax, mic_positions)

                # Display output shape
                self.output_label.config(text=f"Calculation complete! Pytotal shape: {Pytotal.shape}")
            except ValueError as e:
                self.output_label.config(text=f"Error: {e}")

    if __name__ == "__main__":
        root = tk.Tk()
        app = PowerCalculationGUI(root)
        root.mainloop()

    return arr1

def show_plotting_options():
     # Clear all widgets from the master window
    for widget in master.winfo_children():
        widget.destroy()
    

    # Display options for plotting after a signal is loaded.
    tk.Label(master, text="What plot would you like?").grid(row=0, column=1)
    tk.Button(master, text="Data", command=lambda: handle_button_action("Data")).grid(row=1, column=1)
    tk.Button(master, text="Spectrogram", command=lambda: handle_button_action("Spectrogram")).grid(row=2, column=1)
    tk.Button(master, text="SEE", command=lambda: SEE()).grid(row=3, column=1)

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
    elif action == "Data":
        Data_plot(Fs, x)
    elif action == "SEE":
        SEE_plot(Fs, x )


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
row_counter = 6
entry_list = []

# Initial interface setup
reset_to_initial_state()

# Run the Tkinter main loop
master.mainloop()