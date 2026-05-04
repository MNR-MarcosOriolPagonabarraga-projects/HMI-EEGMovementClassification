import os
import re
import customtkinter as ctk
from tkinter import filedialog
import mne

from src.load_data import EEGMatLoader
from src.pipeline import EEGPreprocessor
from src.config import MOTOR_CHANNELS

# --- UI Theme Configuration ---
ctk.set_appearance_mode("dark")  # "dark", "light", or "system"
ctk.set_default_color_theme("blue")  

class ModernEEGVisualizer(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NeuroPipeline: BCI Visualizer")
        self.geometry("400x380")
        self.resizable(False, False)
        
        self.loader = EEGMatLoader(data_root="data/original")
        self.raw = None 
        
        # --- UI Layout ---
        self.grid_columnconfigure(0, weight=1)

        # Title Label
        self.lbl_title = ctk.CTkLabel(self, text="EEG Preprocessing Pipeline", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Status Label
        self.lbl_status = ctk.CTkLabel(self, text="No data loaded", text_color="gray")
        self.lbl_status.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Load Button
        self.btn_load = ctk.CTkButton(self, text="Load .mat File", command=self.load_file, fg_color="#2b7b5c", hover_color="#1e5c44")
        self.btn_load.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # Switches (Modern Checkboxes)
        self.var_filter = ctk.BooleanVar(value=True)
        self.sw_filter = ctk.CTkSwitch(self, text="Apply 8-30Hz Causal Filter", variable=self.var_filter)
        self.sw_filter.grid(row=3, column=0, padx=40, pady=10, sticky="w")
        
        self.var_car = ctk.BooleanVar(value=True)
        self.sw_car = ctk.CTkSwitch(self, text="Apply Common Average Reference", variable=self.var_car)
        self.sw_car.grid(row=4, column=0, padx=40, pady=10, sticky="w")

        # Plot Button
        self.btn_plot = ctk.CTkButton(self, text="Visualize Data", command=self.plot_data, state="disabled")
        self.btn_plot.grid(row=5, column=0, padx=20, pady=(30, 20), sticky="ew")


    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if not filepath: return
        
        # Extract filename to parse Subject and Run using Regex
        filename = os.path.basename(filepath)
        match = re.search(r"ME_S(\d+)_r(\d+)\.mat", filename, re.IGNORECASE)
        
        if not match:
            self.lbl_status.configure(text="Error: Invalid filename format.", text_color="red")
            return
            
        subject_id, run_id = int(match.group(1)), int(match.group(2))
        self.lbl_status.configure(text=f"Loading Sub {subject_id} | Run {run_id}...", text_color="orange")
        self.update() # Force UI update

        try:
            # Load the data using your custom loader
            self.raw = self.loader.load_run(subject=subject_id, run=run_id, sfreq=512.0)
            
            # Keep only the motor channels right away to save memory
            # self.raw.pick(MOTOR_CHANNELS)
            
            # Map the ugly string codes to readable markers
            event_mapping = {
                '1536': 'Elbow Flexion', 
                '1537': 'Elbow Extension', 
                '1538': 'Supination',
                '1539': 'Pronation',
                '1540': 'Hand Close',
                '1541': 'Hand Open',
                '1542': 'Rest'
            }
            # Rename annotations if they exist in the raw file
            current_annots = self.raw.annotations.description
            rename_dict = {k: v for k, v in event_mapping.items() if k in current_annots}
            if rename_dict:
                self.raw.annotations.rename(rename_dict)

            self.lbl_status.configure(text=f"Loaded: {filename}", text_color="#34c759")
            self.btn_plot.configure(state="normal")
            
        except Exception as e:
            self.lbl_status.configure(text=f"Failed to load: {e}", text_color="red")

    def plot_data(self):
        if self.raw is None: return
        
        self.lbl_status.configure(text="Processing pipeline...", text_color="orange")
        self.update()
        
        # 1. Instantiate the pipeline with UI switch states
        pipeline = EEGPreprocessor(
            apply_filter=self.var_filter.get(),
            apply_car=self.var_car.get()
        )
        
        # 2. Process the data
        processed_raw = pipeline.process(self.raw)
        
        self.lbl_status.configure(text="Plotting...", text_color="orange")
        self.update()
        
        # 3. Launch MNE's interactive Qt viewer
        processed_raw.plot(
            block=True, 
            duration=10.0,         # STRICTLY limits the view to 10 seconds (efficient chunking)
            n_channels=len(30), # Fits all motor channels on screen
            scalings=dict(eeg=40e-6), # Standardize voltage scale (40 microvolts)
            clipping='clamp',      # Prevents wild artifact spikes from ruining the view
            title="Interactive EEG Viewer (MNE Qt)"
        )
        
        self.lbl_status.configure(text="Ready.", text_color="#34c759")

if __name__ == "__main__":
    # Ensure MNE uses the high-performance Qt backend
    mne.viz.set_browser_backend('qt') 
    
    app = ModernEEGVisualizer()
    app.mainloop()