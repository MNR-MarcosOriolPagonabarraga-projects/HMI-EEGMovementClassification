from pathlib import Path

# DATASET
DATA_ROOT = Path("data/original")
OUTPUT_DIR = Path("data/processed")
TRAIN_OUTPUT = OUTPUT_DIR / "dataset_train.npz"
TEST_OUTPUT = OUTPUT_DIR / "dataset_test.npz"

# PREPROCESSING
RECORDING_SFREQ = 512.0  # EEGLAB .mat sample rate; event latencies are in these samples
SFREQ = 256.0  # after EEGMatLoader resampling; epochs and processed .npz
EPOCH_TMIN = -0.5
EPOCH_TMAX = 3
BANDPASS_L_FREQ = 3
BANDPASS_H_FREQ = 30
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
    'FCz', 'C3', 'C4', 'Cz'
]

# TRAINING
TRAIN_SIZE = 0.85
TEST_SIZE = 0.15

# CSP + classifier
CSP_N_COMPONENTS = 6
CSP_REG = None