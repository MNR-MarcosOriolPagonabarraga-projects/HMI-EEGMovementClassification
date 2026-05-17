import sys
import os
import re
import mne
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QTreeView, QPushButton, QCheckBox, 
                             QLabel, QSplitter, QDoubleSpinBox, QSpinBox, 
                             QComboBox, QFormLayout, QGroupBox, QScrollArea)
from PyQt6.QtGui import QFileSystemModel, QFont
from PyQt6.QtCore import Qt

from src.load_data import EEGMatLoader
from src.pipeline import EEGPreprocessor
from src.config import MOTOR_CHANNELS

class UnifiedEEGVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NeuroPipeline: Unified BCI Visualizer")
        self.resize(1300, 800) 
        
        self.data_root = "data/original"
        self.loader = EEGMatLoader(data_root=self.data_root, channels=MOTOR_CHANNELS)
        self.raw = None
        self.current_plot_widget = None

        # --- Main Layout Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Main Splitter (Left Column vs Right Graph)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # Left Column Splitter (File Tree vs Controls)
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.left_splitter)

        # ==========================================
        # TOP-LEFT: FILE EXPLORER
        # ==========================================
        self.tree_panel = QWidget()
        tree_layout = QVBoxLayout(self.tree_panel)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_explorer = QLabel("Dataset Explorer")
        lbl_explorer.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        tree_layout.addWidget(lbl_explorer)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.data_root)
        self.file_model.setNameFilters(["*.mat"])
        self.file_model.setNameFilterDisables(False) 

        self.tree = QTreeView()
        self.tree.setModel(self.file_model)
        self.tree.setRootIndex(self.file_model.index(self.data_root))
        for i in range(1, 4): self.tree.hideColumn(i)
        self.tree.doubleClicked.connect(self.on_file_double_clicked)
        
        tree_layout.addWidget(self.tree)
        self.left_splitter.addWidget(self.tree_panel)

        # ==========================================
        # BOTTOM-LEFT: PREPROCESSING CONTROLS
        # ==========================================
        self.control_panel = QWidget()
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(0, 10, 0, 0)

        lbl_controls = QLabel("Preprocessing Pipeline")
        lbl_controls.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        control_layout.addWidget(lbl_controls)

        self.lbl_status = QLabel("Ready. Select a file.")
        self.lbl_status.setStyleSheet("color: gray; margin-bottom: 5px;")
        self.lbl_status.setWordWrap(True)
        control_layout.addWidget(self.lbl_status)

        # Scroll area in case controls get too tall
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0,0,10,0)

        # --- Filter Group ---
        grp_filter = QGroupBox("Spectral Filtering")
        form_filter = QFormLayout(grp_filter)
        
        self.chk_bp = QCheckBox("Causal Bandpass Filter")
        self.chk_bp.setChecked(True)
        self.spin_lfreq = QDoubleSpinBox(); self.spin_lfreq.setRange(0.1, 100.0); self.spin_lfreq.setValue(0.3)
        self.spin_hfreq = QDoubleSpinBox(); self.spin_hfreq.setRange(1.0, 200.0); self.spin_hfreq.setValue(3)
        self.chk_bp.toggled.connect(self.spin_lfreq.setEnabled)
        self.chk_bp.toggled.connect(self.spin_hfreq.setEnabled)
        
        form_filter.addRow(self.chk_bp)
        form_filter.addRow("Low Cutoff (Hz):", self.spin_lfreq)
        form_filter.addRow("High Cutoff (Hz):", self.spin_hfreq)

        self.chk_notch = QCheckBox("Notch Filter (Powerline)")
        self.chk_notch.setChecked(False)
        self.combo_notch = QComboBox()
        self.combo_notch.addItems(["50 Hz (EU)", "60 Hz (US)"])
        self.combo_notch.setEnabled(True)
        self.chk_notch.toggled.connect(self.combo_notch.setEnabled)

        form_filter.addRow(self.chk_notch)
        form_filter.addRow("Frequency:", self.combo_notch)
        scroll_layout.addWidget(grp_filter)

        # --- Spatial & Resampling Group ---
        grp_other = QGroupBox("Spatial & Sampling")
        form_other = QFormLayout(grp_other)

        self.chk_car = QCheckBox("Apply Common Average Reference (CAR)")
        self.chk_car.setChecked(True)
        form_other.addRow(self.chk_car)

        self.chk_resample = QCheckBox("Downsample Data")
        self.chk_resample.setChecked(False)
        self.spin_resample = QSpinBox(); self.spin_resample.setRange(50, 500); self.spin_resample.setValue(256)
        self.spin_resample.setEnabled(False)
        self.chk_resample.toggled.connect(self.spin_resample.setEnabled)
        
        form_other.addRow(self.chk_resample)
        form_other.addRow("Target Rate (Hz):", self.spin_resample)
        scroll_layout.addWidget(grp_other)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        control_layout.addWidget(scroll)

        self.btn_plot = QPushButton("Apply & Visualize")
        self.btn_plot.setEnabled(False)
        self.btn_plot.setStyleSheet("padding: 10px; margin-top: 10px; font-weight: bold; background-color: #0e639c;")
        self.btn_plot.clicked.connect(self.process_and_plot)
        control_layout.addWidget(self.btn_plot)

        self.left_splitter.addWidget(self.control_panel)

        # ==========================================
        # RIGHT PANEL: VISUALIZER
        # ==========================================
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)

        self.placeholder_lbl = QLabel("Visualizer Blank\n\nDouble-click a .mat file to load.")
        self.placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_lbl.setStyleSheet("color: #666666; font-size: 18px;")
        self.right_layout.addWidget(self.placeholder_lbl)

        self.main_splitter.addWidget(self.right_panel)

        # Set splitter sizes
        self.left_splitter.setSizes([400, 400]) # 50/50 split on the left
        self.main_splitter.setSizes([350, 950]) # Left column width vs Graph width

    # --- Methods ---

    def on_file_double_clicked(self, index):
        if self.file_model.isDir(index): return 
        
        filepath = self.file_model.filePath(index)
        filename = os.path.basename(filepath)
        
        match = re.search(r"ME_S(\d+)_r(\d+)\.mat", filename, re.IGNORECASE)
        if not match:
            self.lbl_status.setText("Error: Invalid filename format.")
            self.lbl_status.setStyleSheet("color: #ff4444;")
            return

        subject_id, run_id = int(match.group(1)), int(match.group(2))
        self.lbl_status.setText(f"Loading Sub {subject_id} | Run {run_id}...")
        self.lbl_status.setStyleSheet("color: #ffaa00;")
        QApplication.processEvents() 

        try:
            self.raw = self.loader.load_run(subject=subject_id, run=run_id)
            
            event_mapping = {
                '1536': 'Elbow Flexion', '1537': 'Elbow Ext', 
                '1538': 'Supination', '1539': 'Pronation',
                '1540': 'Hand Close', '1541': 'Hand Open', '1542': 'Rest'
            }
            current_annots = self.raw.annotations.description
            rename_dict = {k: v for k, v in event_mapping.items() if k in current_annots}
            if rename_dict:
                self.raw.annotations.rename(rename_dict)

            self.lbl_status.setText(f"Loaded: {filename}\nReady.")
            self.lbl_status.setStyleSheet("color: #00cc66;")
            self.btn_plot.setEnabled(True)
            
            self.process_and_plot()

        except Exception as e:
            self.lbl_status.setText(f"Failed to load: {e}")
            self.lbl_status.setStyleSheet("color: #ff4444;")

    def process_and_plot(self):
        if self.raw is None: return

        self.lbl_status.setText("Processing pipeline...")
        self.lbl_status.setStyleSheet("color: #ffaa00;")
        QApplication.processEvents()

        pipeline = EEGPreprocessor(
            apply_filter=self.chk_bp.isChecked(),
            l_freq=self.spin_lfreq.value(),
            h_freq=self.spin_hfreq.value(),
            apply_notch=self.chk_notch.isChecked(),
            apply_car=self.chk_car.isChecked(),
            apply_resample=self.chk_resample.isChecked(),
            resample_freq=self.spin_resample.value()
        )
        
        processed_raw = pipeline.process(self.raw)

        self.lbl_status.setText("Generating Plot...")
        QApplication.processEvents()

        if self.placeholder_lbl is not None:
            self.placeholder_lbl.deleteLater()
            self.placeholder_lbl = None

        if self.current_plot_widget is not None:
            self.current_plot_widget.close()
            self.right_layout.removeWidget(self.current_plot_widget)
            self.current_plot_widget.deleteLater()

        self.current_plot_widget = processed_raw.plot(
            show=False,             
            duration=10.0,         
            n_channels=60,            
            scalings=dict(eeg=40e-6), 
            clipping='clamp',
        )
        
        self.right_layout.addWidget(self.current_plot_widget)
        self.lbl_status.setText("Viewing Data.")
        self.lbl_status.setStyleSheet("color: #00cc66;")

# --- Dark Mode Styling ---
DARK_STYLESHEET = """
QMainWindow { background-color: #1e1e1e; color: #ffffff; }
QWidget { background-color: #1e1e1e; color: #ffffff; font-family: Arial; }
QTreeView { background-color: #252526; border: none; outline: none; }
QTreeView::item:selected { background-color: #094771; }
QGroupBox { font-weight: bold; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 15px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; top: -5px; }
QPushButton { border-radius: 4px; color: white; }
QPushButton:hover { background-color: #1177bb; }
QPushButton:disabled { background-color: #333333; color: #777777; }
QSplitter::handle { background-color: #333333; }
QDoubleSpinBox, QSpinBox, QComboBox { background-color: #333333; color: white; border: 1px solid #555; padding: 3px; }
QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled { background-color: #222222; color: #777; }
"""

if __name__ == "__main__":
    mne.viz.set_browser_backend('qt') 
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    
    window = UnifiedEEGVisualizer()
    window.show() 
    sys.exit(app.exec())