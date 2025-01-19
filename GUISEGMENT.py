import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
import os
import samplerate
# from Localization3D.wavaudioread import wavaudioread
from scipy.signal import butter, spectrogram
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft, ifft
from module2.Functionsmodule1 import peaks_baby, Impulse, zero_pad_arrays, h_combine, input_signal, bpm_more_signals, ch3, SEE_plot, Data_plot, plot_spectrogram, cutting, cutting_real1, cuttingarray
from tkinter import ttk
from Localization3D.Functions import music_z, create_points, narrowband_Rx2



# fs=48000
def wavaudioread(filename, fs):
    fs_wav, y_wav = wavfile.read(filename)
    y = samplerate.resample(y_wav, fs / fs_wav, "sinc_best")

    return y

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
    global info_label, signal_own, folder_6_array, finalarr, entry_list # Use global so we can access it for destruction later

    # Clear the window by destroying all widgets
    for widget in master.winfo_children():
        widget.destroy()

    finalarr = []
    entry_list = []
    d=0
    h=0

    # Recreate initial interface
    info_label = tk.Label(master, text="What example data would you like to use?")
    info_label.grid(row=0, column=1)
    info2_label = tk.Label(master, text="Your own signal. Copy relative path and change \ to /")
    info2_label.grid(row=3,column=1)
    info3_label = tk.Label(master, text="For localization please fill in a folder of 6 microphone sounds")
    info3_label.grid(row=5, column=1)

    tk.Button(master, text="Signal 1", command=lambda: data_chooser(1)).grid(row=1, column=1, padx=10, pady=2)
    tk.Button(master, text="Signal 2", command=lambda: data_chooser(2)).grid(row=2, column=1, padx=10, pady=2)
    tk.Button(master,text="Confirm", command = signal_getten).grid(row=4, column=2,padx=10, pady=10, sticky="w")
    tk.Button(master,text="Confirm", command = folder_getten).grid(row=6, column=2,padx=10, pady=10, sticky="w")
    
    signal_own = tk.Entry(master, width=40)
    signal_own.grid(row=4, column=1, padx=10, pady=2)

    folder_6_array = tk.Entry(master, width=40)
    folder_6_array.grid(row=6, column=1, padx=10, pady=2)






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



def folder_getten():
    global audio_matrix, Fs
    folder = folder_6_array.get()
    if folder:
        Fs = 48000
        # Get list of all .wav files in the folder
        wav_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.WAV')]

        # Initialize an empty list to store signals
        audio_signals = []

        # Read each .wav file
        for wav_file in wav_files:
            signal = wavaudioread(wav_file, Fs)  # Read file
            audio_signals.append(signal)

        # Convert to matrix (numpy array)
        audio_matrix = np.array(audio_signals).T
        localization_gui()
    else:
        print("no worky")


    


def localization_gui():
    global first_entry, second_entry, finalarr, entry_list,plot, cut, hdistance, dheight

    for widget in master.winfo_children():
        widget.destroy()

    finalarr = []
    entry_list = []
    # d=0
    # h=0

    # Add new label and entry for minimal distance
    tk.Label(master, text="Set variables").grid(row=0, column=0, padx=10, pady=2)
    tk.Label(master, text="Distance Threshold").grid(row=1, column=0, padx=10, pady=2)
    tk.Label(master, text="Peaks Threshold").grid(row=2, column=0, padx=10, pady=2)
    

    # Create and store the Entry widget reference
    hdistance = tk.Entry(master)
    hdistance.grid(row=1, column=1, padx=10, pady=2)

    dheight = tk.Entry(master)
    dheight.grid(row=2, column=1, padx=10, pady=2)
    
    # Add a confirm button to retrieve the entry value
    tk.Button(master, text="Confirm", command=hdlocal_getten).grid(row=2, column=2, columnspan=2, pady=10, sticky="w")
    cut = tk.Button(master, text="I want to cut", command=i_want_to_cut)
    cut.grid(row=3, column=0, columnspan=2, pady=10)
    plot=tk.Button(master, text="Plot", command=lambda:reinfunctie(1))
    plot.grid(row=3, column=1, columnspan=2, pady=10)

    # Add a "Go Back" button
    tk.Button(master, text="Go Back", command=reset_to_initial_state).grid(row=0, column=2)
    tk.Button(master, text="localization", command=localizing()).grid(row=1 , column=3 , padx=10 , pady=2)




def localizing():
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


    
   

def i_want_to_cut():
    global first_entry,second_entry
    plot.destroy()
    cut.destroy()

    tk.Label(master, text="Cut sections in time intervals").grid(row=3, column=0, columnspan=3, pady=2)
    tk.Label(master, text="Start Time").grid(row=5, column=0, padx=5, pady=2, sticky="e")
    tk.Label(master, text="End Time").grid(row=5, column=1, padx=5, pady=2, sticky="w")

    
    
    tk.Button(master, text="Add index", command=Add_index).grid(row=4, column=0, columnspan=2, pady=2)
    tk.Button(master, text="Confirm", command=cutting_arr).grid(row=row_counter + 5, column=0, padx=10, pady=2, sticky="e")
    tk.Button(master, text="Plot cutted signal", command=lambda: reinfunctie(2)).grid(row=row_counter + 5, column=1, padx=10, pady=2, sticky="w")
    

    first_entry = tk.Entry(master)
    first_entry.grid(row=6, column=0, padx=5, pady=2)
    entry_list.append(first_entry)
    second_entry = tk.Entry(master)
    second_entry.grid(row=6, column=1, padx=5, pady=2)
    entry_list.append(second_entry)

def reinfunctie(opdecut):
    global Fs, finalarr, audio_matrix
    if opdecut == 1:
        # cuttingarray(Fs, finalarr, audio_matrix)
        period=1/Fs
        y, y2, ynorm , ynorm2, piecesS1, piecesS2, upperlimitS1,  lowerlimitS1, upperlimitS2, lowerlimitS2= cutting_real1(Fs, audio_matrix, hloc, dloc)
        plt.subplot(411)
        t= np.linspace(0,period*len(y),len(y))
        plt.plot(t,ynorm)

        plt.subplot(412)
        t2= np.linspace(0,period*len(y2),len(y2))
        for i in range(len(upperlimitS1)):
            plt.axvline(x=upperlimitS1[i]/Fs, color='r', linewidth=0.7, linestyle='--')
            plt.axvline(x=lowerlimitS1[i]/Fs, color='r', linewidth=0.7, linestyle='--')
        for i in range(len(upperlimitS2)):
            plt.axvline(x=upperlimitS2[i]/Fs, color='b', linewidth=0.7, linestyle='--')
            plt.axvline(x=lowerlimitS2[i]/Fs, color='b', linewidth=0.7, linestyle='--')
        plt.plot(t2,ynorm2)

        plt.subplot(413)
        t3 = np.linspace(0,period*len(piecesS1),len(piecesS1)) 
        plt.plot(t3, piecesS1)

        plt.subplot(414)
        t4 = np.linspace(0,period*len(piecesS2),len(piecesS2)) 
        plt.plot(t4,piecesS2)
        plt.tight_layout()
        # plt.plot(audio_matrix2)
        plt.show()

    
    elif opdecut == 2:
        

        audio_matrix = cuttingarray(Fs, finalarr, audio_matrix)
        period=1/Fs
        y, y2, ynorm, ynorm2, piecesS1, piecesS2, upperlimitS1, lowerlimitS1, upperlimitS2, lowerlimitS2 = cutting_real1(Fs, audio_matrix, hloc, dloc)
        plt.subplot(411)
        t= np.linspace(0,period*len(y),len(y))
        plt.plot(t,ynorm)

        plt.subplot(412)
        t2= np.linspace(0,period*len(y2),len(y2))
        for i in range(len(upperlimitS1)):
            plt.axvline(x=upperlimitS1[i]/Fs, color='r', linewidth=0.7, linestyle='--')
            plt.axvline(x=lowerlimitS1[i]/Fs, color='r', linewidth=0.7, linestyle='--')
        for i in range(len(upperlimitS2)):
            plt.axvline(x=upperlimitS2[i]/Fs, color='b', linewidth=0.7, linestyle='--')
            plt.axvline(x=lowerlimitS2[i]/Fs, color='b', linewidth=0.7, linestyle='--')
        plt.plot(t2,ynorm2)

        plt.subplot(413)
        t3 = np.linspace(0,period*len(piecesS1),len(piecesS1)) 
        plt.plot(t3, piecesS1)

        plt.subplot(414)
        t4 = np.linspace(0,period*len(piecesS2),len(piecesS2)) 
        plt.plot(t4,piecesS2)
        plt.tight_layout()
        # plt.plot(audio_matrix2)
        plt.show()



 
def distance_getten():
    global d, h
    # Retrieve the value from the Entry widget
    try:
        distance_value = distance.get()  # Use the global `distance` reference
        d = int(distance_value)  # Convert the input to an integer
        
    except ValueError:
        print("Please enter a valid number.")

    try:
        height_value = height.get()  # Get the height value from the Entry widget
        h = float(height_value)  # Convert to integer for height
    except ValueError:
        print("Please enter a valid number for height.")
        return  # If the height is invalid, stop the function
    
    peaks_baby(Fs,x,d, h )

def hdlocal_getten():
    global dloc, hloc
    # Retrieve the value from the Entry widget
    try:
        hdistance_value = hdistance.get()  # Use the global `distance` reference
        dloc = int(hdistance_value)  # Convert the input to an integer
        
    except ValueError:
        print("Please enter a valid number.")

    try:
        dheight_value = dheight.get()  # Get the height value from the Entry widget
        hloc = float(dheight_value)  # Convert to integer for height
    except ValueError:
        print("Please enter a valid number for height.")
        return  # If the height is invalid, stop the function
    
    

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
    global xnorm,finalarr
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
    # xnorm = cutting(x, Fs, finalarr)
    
    return xnorm

def plot_new_xnorm():
    xnorm = cutting(x, Fs, finalarr)
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
audio_matrix = None
Fs = None
x = None
d = None
h = None
dloc=None
hloc=None
so= None
xnorm= None
row_counter = 7
entry_list = []

# Initial interface setup
reset_to_initial_state()

# Run the Tkinter main loop
master.mainloop()