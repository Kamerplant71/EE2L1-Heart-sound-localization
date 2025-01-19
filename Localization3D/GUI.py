import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from wavaudioread import wavaudioread
from tkinter import ttk
from Functions import music_z, create_points, narrowband_Rx2
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
    
signal = wavaudioread("recordings\\recording_dual_channel_white_noise.wav", fs)
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
