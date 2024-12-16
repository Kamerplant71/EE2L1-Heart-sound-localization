import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import soundfile as sf
import sounddevice as sd
import numpy as np
import threading

class AudioRecorderApp:
    def __init__(self, root, channels=1):
        self.root = root
        self.root.title("Audio Recorder App")
        self.root.geometry("400x350")

        self.channels = channels  # Number of channels for recording
        self.reference_signal = None
        self.reference_sample_rate = 48000
        self.recording = None
        self.recording_sample_rate = 48000  # Default sample rate

        self.audio_devices = sd.query_devices()
        self.speaker_device = None
        self.microphone_device = None

        self.is_recording = False  # Flag to control the recording
        self.stream = None  # The sounddevice stream object

        # Centralized UI layout
        self.create_widgets()

    def create_widgets(self):
        # Create a frame for all controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(expand=True)

        # Upload Reference Signal button
        self.upload_ref_button = tk.Button(control_frame, text="Upload Reference Signal", command=self.upload_reference_signal)
        self.upload_ref_button.pack(pady=10)

        # Select Speaker Source (Output Device)
        tk.Label(control_frame, text="Select Speaker Source:").pack(pady=5)
        self.speaker_selector = ttk.Combobox(control_frame, state="readonly", values=[d['name'] for d in self.audio_devices if d['max_output_channels'] > 0])
        self.speaker_selector.pack(pady=5)

        # Select Microphone Source (Input Device)
        tk.Label(control_frame, text="Select Microphone Source:").pack(pady=5)
        self.microphone_selector = ttk.Combobox(control_frame, state="readonly", values=[d['name'] for d in self.audio_devices if d['max_input_channels'] > 0])
        self.microphone_selector.pack(pady=5)

        # Start Recording button
        self.start_recording_button = tk.Button(control_frame, text="Start Recording", state=tk.DISABLED, command=self.start_recording)
        self.start_recording_button.pack(pady=10)

        # Stop Recording button
        self.stop_recording_button = tk.Button(control_frame, text="Stop Recording", state=tk.DISABLED, command=self.stop_recording)
        self.stop_recording_button.pack(pady=10)

        # Save Recording button
        self.save_recording_button = tk.Button(control_frame, text="Save Recording", state=tk.DISABLED, command=self.save_recording)
        self.save_recording_button.pack(pady=10)

    def upload_reference_signal(self):
        # Let user select a reference signal file
        file_path = filedialog.askopenfilename(title="Select Reference Audio", filetypes=[("Audio Files", "*.wav *.flac")])
        if file_path:
            try:
                self.reference_signal, self.reference_sample_rate = sf.read(file_path)
                messagebox.showinfo("Success", "Reference signal loaded successfully!")
                self.start_recording_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio file: {str(e)}")

    def start_recording(self):
        # Get selected speaker and microphone
        speaker_name = self.speaker_selector.get()
        microphone_name = self.microphone_selector.get()

        # Find the corresponding indices for selected devices
        for idx, device in enumerate(self.audio_devices):
            if device['name'] == speaker_name and device['max_output_channels'] > 0:
                self.speaker_device = idx
            if device['name'] == microphone_name and device['max_input_channels'] > 0:
                self.microphone_device = idx

        if self.speaker_device is None or self.microphone_device is None:
            messagebox.showerror("Error", "Please select both speaker and microphone devices.")
            return

        # Start recording and playback simultaneously
        self.is_recording = True
        self.stop_recording_button.config(state=tk.NORMAL)
        self.start_recording_button.config(state=tk.DISABLED)
        
        # Start a new thread for recording and playback to avoid freezing the UI
        threading.Thread(target=self.record_and_playback).start()

    def record_and_playback(self):
        try:
            # Initialize an empty array for recording
            self.recording = np.empty((0, self.channels), dtype=np.float32)
            
            # Callback function to handle audio input (recording)
            def audio_callback(indata, frames, time, status):
                if self.is_recording:
                    self.recording = np.append(self.recording, indata, axis=0)

            # Create a sounddevice input stream for recording
            self.stream = sd.InputStream(device=self.microphone_device, channels=self.channels, samplerate=self.recording_sample_rate, callback=audio_callback)
            
            # Start recording and playback simultaneously
            with self.stream:
                sd.play(self.reference_signal, self.reference_sample_rate, device=self.speaker_device)
                while self.is_recording:
                    sd.sleep(100)  # Poll every 100ms to allow stopping

        except Exception as e:
            messagebox.showerror("Error", f"Recording/Playback failed: {str(e)}")

    def stop_recording(self):
        # Stop the recording process
        self.is_recording = False
        self.stop_recording_button.config(state=tk.DISABLED)
        self.save_recording_button.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Recording stopped successfully!")

    def save_recording(self):
        # Save the recorded audio to a file
        file_path = filedialog.asksaveasfilename(title="Save Recording", defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                sf.write(file_path, self.recording, self.recording_sample_rate)
                messagebox.showinfo("Success", "Recording saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save recording: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root, channels=8)  # Set the number of channels to 2 for stereo recording
    root.mainloop()

