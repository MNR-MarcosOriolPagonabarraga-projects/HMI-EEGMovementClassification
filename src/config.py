from pathlib import Path

# DATASET
DATA_ROOT = Path("data/original")
OUTPUT_DIR = Path("data/processed")
TRAIN_OUTPUT = OUTPUT_DIR / "dataset_train.npz"
TEST_OUTPUT = OUTPUT_DIR / "dataset_test.npz"

# PREPROCESSING
RECORDING_SFREQ = 512.0  # EEGLAB .mat sample rate; event latencies are in these samples
SFREQ = 128.0  # after EEGMatLoader resampling; epochs and processed .npz
EPOCH_TMIN = -0.5
EPOCH_TMAX = 2
BANDPASS_L_FREQ = 0.5
BANDPASS_H_FREQ = 40
NOTCH_FREQ = 50.0
REJECT_EEG_UV = 200.0

# LABELS
CLASS_NAMES: tuple[str, ...] = (
    "elbow_flex_ext",
    "hand_open_close",
    "forearm_sup_pron",
    "rest",
)

EVENT_ID: dict[str, int] = {
    "1536": 0,
    "1537": 0,  # elbow flexion / extension
    "1540": 1,
    "1541": 1,  # hand close / open
    "1538": 2,
    "1539": 2,  # supination / pronation
    "1542": 3,  # rest
}

MOTOR_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
]

# TRAINING
TRAIN_SIZE = 0.85
TEST_SIZE = 0.15

# CSP + classifier
CSP_N_COMPONENTS = 6
CSP_REG = None

TUNE_WINDOWS_SPEC = "-1,1;-0.5,3;-0.5,2;-1,2.5"
TUNE_BANDS_SPEC = "0.3,3;3,30;8,30;8,70;0.3,70"
TUNE_RANDOM_STATE = 42
# Summary JSON path relative to repo root unless absolute; None skips writing JSON.
TUNE_OUTPUT_JSON = Path("results/tune_csp_grid.json")