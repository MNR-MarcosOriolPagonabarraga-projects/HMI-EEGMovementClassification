import re
import mne
import numpy as np
from pathlib import Path
from scipy.io.matlab import mat_struct
from sklearn.model_selection import train_test_split

from .config import (
    BANDPASS_H_FREQ, BANDPASS_L_FREQ,
    CLASS_NAMES, EVENT_ID,
    EPOCH_TMAX, EPOCH_TMIN,
    REJECT_EEG_UV
)

# LOAD DATA

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
        
        # Check if the 3rd column exists AND contains a movement onset (> 0)
        if events.shape[1] > 2 and float(row[2]) > 0:
            actual_lat = float(row[2])
        else:
            # Fallback to the cue latency (e.g., for the Rest class where row[2] == 0)
            actual_lat = cue_lat
            
        # Convert 1-based sample index to seconds
        onset_s = (actual_lat - 1.0) / sfreq
        
        onsets.append(onset_s)
        durations.append(0.0) # Point events, no duration needed for epoching
        descs.append(str(typ))

    return mne.Annotations(
        onset=onsets, duration=durations, description=descs,
    )

# BUILD DATASET

def discover_subject_ids(original_root: Path) -> list[int]:
    """List subject integer ids whose folders contain at least one ``ME_*.mat``."""
    ids: list[int] = []
    if not original_root.is_dir():
        raise FileNotFoundError(
            f"Expected raw data directory at {original_root} (does not exist)."
        )

    for d in sorted(original_root.iterdir()):
        if not d.is_dir():
            continue
        if not any(d.glob("ME_*.mat")):
            continue
        name = d.name.upper()
        if name.startswith("S"):
            suffix = name[1:]
            if suffix.isdigit():
                ids.append(int(suffix))

    ids = sorted(set(ids))
    if not ids:
        raise FileNotFoundError(
            f"No subject folders with ME_*.mat under {original_root}"
        )
    return ids


def parse_run_filename(stem: str) -> tuple[int, int] | None:
    fname_re = re.compile(r"ME_S(?P<sub>\d+)_r(?P<run>\d+)\.mat$", re.IGNORECASE)

    m = fname_re.match(stem + ".mat")
    if not m:
        return None
    return int(m.group("sub")), int(m.group("run"))


def discover_run_indices(loader, subject: str | int) -> list[int]:
    """Return sorted run indices for one subject."""
    folder_name, subject_num = _subject_folder_and_id(subject)
    sub_dir = loader.data_root / folder_name
    if not sub_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {sub_dir}")

    runs: list[int] = []
    for path in sorted(sub_dir.glob("*.mat")):
        parsed = parse_run_filename(path.stem)
        if parsed is None:
            continue
        file_sub_id, run_id = parsed
        if file_sub_id != subject_num:
            continue
        runs.append(run_id)

    if not runs:
        raise FileNotFoundError(f"No ME_* runs found under {sub_dir}")

    return sorted(set(runs))


def _subject_folder_and_id(subject: str | int) -> tuple[str, int]:
    """Match ``EEGMatLoader`` naming: folder ``S{n}``, filenames ``ME_S{n:02d}_``. """
    s = str(subject).strip().upper()
    if s.startswith("S"):
        s = s[1:]
    n = int(s)
    return f"S{n}", n


def preprocess_raw_inplace(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """EEG only → CAR → causal 8–30 Hz bandpass (mutates ``raw``)."""
    raw.pick('eeg')
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(
        l_freq=BANDPASS_L_FREQ,
        h_freq=BANDPASS_H_FREQ,
        fir_design="firwin",
        phase="minimum",
        verbose=False,
    )
    return raw


def cue_events_and_epoch(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Cue-locked epochs, pre-cue baseline, constructor peak-to-peak rejection."""
    events, event_id_sel = mne.events_from_annotations(raw, event_id=EVENT_ID)

    baseline = (None, 0)
    reject = dict(eeg=REJECT_EEG_UV * 1e-6)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_sel,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=baseline,
        picks="eeg",
        reject=reject,
        preload=True,
        verbose=False,
    )
    epochs.drop_bad()
    return epochs


def epochs_to_XY(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """``X``: (trials x channels x samples); ``y``: meta-class ids 0..3."""
    X = epochs.get_data(copy=True).astype(np.float32, copy=False)
    y = epochs.events[:, 2].astype(np.int64, copy=False)
    return X, y


def build_subject_epochs(
    loader,
    subject: str | int,
    *,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """All runs for one subject → concatenated ``X``, ``y`` plus ``ch_names``."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ch_names_ref: list[str] | None = None

    for run in discover_run_indices(loader, subject):
        raw = loader.load_run(subject, run, sfreq=sfreq)
        preprocess_raw_inplace(raw)
        epochs = cue_events_and_epoch(raw)

        if len(epochs) == 0:
            continue

        names = epochs.ch_names.copy()
        if ch_names_ref is None:
            ch_names_ref = names
        elif names != ch_names_ref:
            raise RuntimeError(
                "EEG channel order mismatch within subject "
                f"{subject}: expected {ch_names_ref[:6]}..., got {names[:6]}..."
            )

        Xi, yi = epochs_to_XY(epochs)
        xs.append(Xi)
        ys.append(yi)

    if not xs or ch_names_ref is None:
        raise RuntimeError(
            f"No usable epochs after preprocessing for subject {subject}"
        )

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), ch_names_ref


def build_pooled_arrays(
    loader,
    *,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack trials from every discovered subject/run into single arrays."""
    subject_ids = discover_subject_ids(loader.data_root)
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    ch_names_global: list[str] | None = None

    for sid in subject_ids:
        tag = _subject_folder_and_id(sid)[0].upper()
        try:
            Xs, ys, chs = build_subject_epochs(loader, sid, sfreq=sfreq)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[build_dataset] skip {tag}: {exc}")
            continue

        if ch_names_global is None:
            ch_names_global = chs
        elif chs != ch_names_global:
            raise RuntimeError(
                f"Channel mismatch for {tag}: first subject had "
                f"{ch_names_global[:6]}..., this has {chs[:6]}..."
            )

        X_chunks.append(Xs)
        y_chunks.append(ys)

    if not X_chunks or ch_names_global is None:
        raise RuntimeError("No pooled data assembled (check logs / raw tree).")

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)

    counts = tuple(int((y == c).sum()) for c in range(len(CLASS_NAMES)))
    print(f"[build_dataset] pooled X{X.shape}; per-class counts {counts}")

    return X, y, ch_names_global


def save_compressed_npz(
    path: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    split_tag: str,
) -> None:
    """Write one split (.npz) with tensors + bookkeeping arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        sfreq=np.array([sfreq], dtype=np.float64),
        ch_names=np.array(ch_names, dtype=object),
        class_names=np.array(CLASS_NAMES, dtype=object),
        split=np.array([split_tag], dtype=object),
        epoch_tmin=np.array([EPOCH_TMIN]),
        epoch_tmax=np.array([EPOCH_TMAX]),
        bandpass_hz=np.array([BANDPASS_L_FREQ, BANDPASS_H_FREQ], dtype=np.float64),
        causal_filter_phase=np.array(["minimum"], dtype=object),
        reject_eeg_uv=np.array([REJECT_EEG_UV]),
    )


def stratified_mix_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 1-test_fraction / test_fraction shuffle split."""
    return train_test_split(
        X,
        y,
        test_size=test_fraction,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )