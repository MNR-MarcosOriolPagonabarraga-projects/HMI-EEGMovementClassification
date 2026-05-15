"""Shared helpers for loading EEGLAB ``*.mat`` structures into MNE."""

import mne
import numpy as np
from scipy.io.matlab import mat_struct


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
