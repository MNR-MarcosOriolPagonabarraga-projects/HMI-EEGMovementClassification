import re
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    BANDPASS_H_FREQ,
    BANDPASS_L_FREQ,
    CLASS_NAMES,
    DATA_ROOT,
    EPOCH_TMAX,
    EPOCH_TMIN,
    EVENT_ID,
    REJECT_EEG_UV,
    SFREQ,
    TEST_OUTPUT,
    TEST_SIZE,
    TRAIN_OUTPUT,
    TRAIN_SIZE,
    MOTOR_CHANNELS
)
from src.load_data import EEGMatLoader
from src.pipeline import EEGPreprocessor

_RUN_STEM_RE = re.compile(r"ME_S(?P<sub>\d+)_r(?P<run>\d+)\.mat$", re.IGNORECASE)


def main() -> None:
    # Discover subjects and runs on disk
    loader = EEGMatLoader(
        data_root=DATA_ROOT,
        channels=MOTOR_CHANNELS,
        target_sfreq=SFREQ,
    )
    subject_ids = _discover_subject_ids(DATA_ROOT)
    print(
        f"[build_dataset] subjects={len(subject_ids)}, "
        f"data_root={DATA_ROOT}\n\tids={subject_ids}"
    )

    # Continuous EEG preprocessing (same class as the visualizer script)
    preprocessor = EEGPreprocessor(
        apply_filter=True,
        l_freq=BANDPASS_L_FREQ,
        h_freq=BANDPASS_H_FREQ,
        apply_notch=True,
        apply_car=False,
        apply_resample=False,
    )

    # Concatenate all trial tensors (trial × channel × time)
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    ch_names_global: list[str] | None = None

    for sid in subject_ids:
        tag = _subject_folder_and_id(sid)[0].upper()
        try:
            Xs, ys, chs = _trials_for_subject(loader, sid, preprocessor)
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

    # 4) Stratified train/test partition (sklearn)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
        shuffle=True,
    )
    train_frac_real = len(y_train) / len(y)
    test_frac_real = len(y_test) / len(y)
    print(
        "[build_dataset] split sizes:"
        f" train {len(y_train)} ({train_frac_real:.3%}),"
        f" test {len(y_test)} ({test_frac_real:.3%});"
        f" target {TRAIN_SIZE:.0%}/{TEST_SIZE:.0%}"
    )

    # 5) Write compressed archives with metadata for training scripts
    _write_split_npz(
        TRAIN_OUTPUT,
        X=X_train,
        y=y_train,
        sfreq=loader.output_sfreq,
        ch_names=ch_names_global,
        split_tag="train",
    )
    _write_split_npz(
        TEST_OUTPUT,
        X=X_test,
        y=y_test,
        sfreq=loader.output_sfreq,
        ch_names=ch_names_global,
        split_tag="test",
    )

    print(
        f"[build_dataset] wrote {TRAIN_OUTPUT} and {TEST_OUTPUT} "
        f"({len(ch_names_global)} EEG channels, sfreq={loader.output_sfreq} Hz)."
    )


def _discover_subject_ids(original_root: Path) -> list[int]:
    """Subject integer ids under ``original_root`` that contain at least one ``ME_*.mat``."""
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


def _subject_folder_and_id(subject: str | int) -> tuple[str, int]:
    """Folder ``S{n}`` and numeric id; matches ``EEGMatLoader`` naming."""
    s = str(subject).strip().upper()
    if s.startswith("S"):
        s = s[1:]
    n = int(s)
    return f"S{n}", n


def _parse_run_stem(stem: str) -> tuple[int, int] | None:
    m = _RUN_STEM_RE.match(stem + ".mat")
    if not m:
        return None
    return int(m.group("sub")), int(m.group("run"))


def _run_indices_for_subject(loader: EEGMatLoader, subject: str | int) -> list[int]:
    folder_name, subject_num = _subject_folder_and_id(subject)
    sub_dir = loader.data_root / folder_name
    if not sub_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {sub_dir}")

    runs: list[int] = []
    for path in sorted(sub_dir.glob("*.mat")):
        parsed = _parse_run_stem(path.stem)
        if parsed is None:
            continue
        file_sub_id, run_id = parsed
        if file_sub_id != subject_num:
            continue
        runs.append(run_id)

    if not runs:
        raise FileNotFoundError(f"No ME_* runs found under {sub_dir}")

    return sorted(set(runs))


def _cue_locked_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    events, event_id_sel = mne.events_from_annotations(raw, event_id=EVENT_ID)
    reject = dict(eeg=REJECT_EEG_UV * 1e-6)
    epochs = mne.Epochs(
        raw,
        events,
        event_id_sel,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=(None, 0),
        picks="eeg",
        reject=reject,
        preload=True,
        verbose=False,
    )
    epochs.drop_bad()
    return epochs


def _epochs_to_arrays(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Extracts data, normalizes per epoch/channel, and shapes for PyTorch."""
    # X shape: (epochs, channels, time)
    X = epochs.get_data(copy=True).astype(np.float32, copy=False)
    y = epochs.events[:, 2].astype(np.int64, copy=False)
    
    # Z-SCORE NORMALIZATION (Per-epoch, Per-channel)
    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)
    
    # Apply normalization
    X_norm = (X - mean) / (std + 1e-8)
    
    return X_norm, y


def _write_split_npz(
    path: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    split_tag: str,
) -> None:
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
        # Updated metadata to reflect the removal of minimum phase
        causal_filter_phase=np.array(["zero"], dtype=object), 
        reject_eeg_uv=np.array([REJECT_EEG_UV]),
        normalization=np.array(["per_epoch_channel_zscore"], dtype=object) # Added tracker
    )

def _trials_for_subject(
    loader: EEGMatLoader,
    subject: str | int,
    preprocessor: EEGPreprocessor,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """All runs for one subject → stacked trial arrays and channel names."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ch_names_ref: list[str] | None = None

    for run in _run_indices_for_subject(loader, subject):
        raw = loader.load_run(subject, run)
        raw.pick("eeg", verbose=False)
        raw = preprocessor.process(raw)
        epochs = _cue_locked_epochs(raw)

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

        Xi, yi = _epochs_to_arrays(epochs)
        xs.append(Xi)
        ys.append(yi)

    if not xs or ch_names_ref is None:
        raise RuntimeError(
            f"No usable epochs after preprocessing for subject {subject}"
        )

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), ch_names_ref


if __name__ == "__main__":
    main()
