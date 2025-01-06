import tkinter as tk
from tkinter import ttk
from Functionsmodule1 import peaks_baby


def handle_action(action=None):
    """Handles both retrieving entry values and button actions."""
    FS = e1.get()
    distance = e2.get()
    if action:
        print(action)
    print(f"FS: {FS}, Distance: {distance}")  # Optional, to debug the retrieved values.

master = tk.Tk()
master.title("Heartbeat")
master.geometry("700x250")

# Create labels and entry
e1 = tk.Entry(master)
e1.grid(row=1, column=1)
e2 = tk.Entry(master)
e2.grid(row=11,column=1)
 
# Information
info=tk.Label(master, text="What is the sample frequency?").grid(row=0,column=1)
info=tk.Label(master, text="What plot would you like?").grid(row=5,column=1)
info=tk.Label(master, text="Minimal distance between the peaks to be?").grid(row=10,column=1)

# Create checkbutton
confirm_button_FS = tk.Button(master, text="Confirm", command=lambda: handle_action()).grid(row=1, column=2)
confirm_button_distance = tk.Button(master, text="Confirm", command=lambda: handle_action()).grid(row=11, column=2)
heartsignal_button = tk.Button(master, text="Heartsignal", command=lambda: handle_action("Heartbeat")).grid(row=6, column=1)
se_button = tk.Button(master, text="Shannon energy", command=lambda: handle_action("Shannon energy")).grid(row=7, column=1)
SEE_button = tk.Button(master, text="SEE", command=lambda: handle_action("SEE")).grid(row=8, column=1)

master.mainloop()