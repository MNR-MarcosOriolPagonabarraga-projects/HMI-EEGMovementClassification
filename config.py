from pathlib import Path

# DATASET
DATA_ROOT = Path("data/original")
OUTPUT_DIR = Path("data/processed")
TRAIN_OUTPUT = OUTPUT_DIR / "dataset_train.npz"
TEST_OUTPUT = OUTPUT_DIR / "dataset_test.npz"

# PREPROCESSING
SFREQ = 512.0
EPOCH_TMIN = -0.5
EPOCH_TMAX = 3.0
BANDPASS_L_FREQ = 8.0
BANDPASS_H_FREQ = 30.0
REJECT_EEG_UV = 150.0

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

# TRAINING
TRAIN_SIZE = 0.85
TEST_SIZE = 0.15 