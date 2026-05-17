import mne
import numpy as np
from scipy.io.matlab import mat_struct
import matplotlib.pyplot as plt


def _chan_label(ch: mat_struct) -> str:
    raw = getattr(ch, "labels", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip() or "unknown"


def _infer_ch_type(name: str) -> str:
    u = name.lower()
    if "eog" in u:
        return "eog"
    if any(
        x in u
        for x in (
            "thumb",
            "index",
            "middle",
            "ring",
            "litte",
            "palm",
            "wrist",
            "hand",
            "elbow",
            "shoulder",
            "grip",
            "gesture",
            "roll",
            "pitch",
            "pos",
        )
    ):
        return "misc"
    return "eeg"


def _xyz_mm(ch: mat_struct) -> tuple[float, float, float] | None:
    def as_float(name: str) -> float | None:
        v = getattr(ch, name, None)
        if v is None:
            return None
        arr = np.asarray(v).squeeze()
        if arr.size == 0:
            return None
        try:
            return float(arr)
        except (TypeError, ValueError):
            return None

    x, y, z = as_float("X"), as_float("Y"), as_float("Z")
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _events_to_annotations(
    events: np.ndarray, sfreq: float
) -> mne.Annotations:
    """Build annotations from the (n, 3) EEGLAB matrix in these files.

    Columns are interpreted based on the specific dataset paradigm:
    Col 1: Event code (type)
    Col 2: Visual cue latency (samples)
    Col 3: Actual movement onset latency (samples). 0 if no movement (Rest).
    """
    if events.ndim != 2 or events.shape[1] < 2:
        return mne.Annotations(onset=[], duration=[], description=[])

    onsets = []
    durations = []
    descs = []

    for row in events:
        typ = int(row[0])
        cue_lat = float(row[1])

        if events.shape[1] > 2 and float(row[2]) > 0:
            actual_lat = float(row[2])
        else:
            actual_lat = cue_lat

        onset_s = (actual_lat - 1.0) / sfreq

        onsets.append(onset_s)
        durations.append(0.0)
        descs.append(str(typ))

    return mne.Annotations(
        onset=onsets, duration=durations, description=descs,
    )


def load_npz_split(path) -> tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float64)
    X_psd = np.asarray(z["X_psd"], dtype=np.float64)
    y = np.asarray(z["y"]).astype(np.int64).ravel()
    meta = {
        "sfreq": float(np.asarray(z["sfreq"]).squeeze()),
        "ch_names": [str(x) for x in np.asarray(z["ch_names"], dtype=object).ravel()],
        "split": str(np.asarray(z["split"]).ravel()[0]),
        "path": path,
    }
    return X, X_psd, y, meta


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    ax1.plot(history['train_loss'], label='Train Loss', color='tab:blue', lw=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='tab:red', lw=2)
    ax1.set_title('CrossEntropy Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', color='tab:blue', lw=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='tab:red', lw=2)
    ax2.axhline(25, color='black', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax2.set_title('Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig = plt.gcf()

    return fig