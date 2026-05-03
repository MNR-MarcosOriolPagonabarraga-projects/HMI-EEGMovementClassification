from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
from mne.io import RawArray
from scipy.io import loadmat
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

    Columns are interpreted as: ``type`` (code), ``latency`` (samples,
    1-based as in EEGLAB), ``end_sample`` (exclusive end sample, or 0 for
    point events).
    """
    if events.ndim != 2 or events.shape[1] < 2:
        return mne.Annotations(onset=[], duration=[], description=[])

    onsets = []
    durations = []
    descs = []
    for row in events:
        typ, lat = int(row[0]), float(row[1])
        end_samp = float(row[2]) if events.shape[1] > 2 else 0.0
        onset_s = (lat - 1.0) / sfreq
        if end_samp > lat:
            dur_s = (end_samp - lat) / sfreq
        else:
            dur_s = 0.0
        onsets.append(onset_s)
        durations.append(dur_s)
        descs.append(str(typ))

    return mne.Annotations(
        onset=onsets, duration=durations, description=descs,
    )


class EEGMatLoader:
    """Read ``ME_{subject}_r{run}.mat`` files under ``data/S{subject}/``."""

    _fname_re = re.compile(r"ME_S(?P<sub>\d+)_r(?P<run>\d+)\.mat$", re.IGNORECASE)

    def __init__(self, data_root: str | Path = "data") -> None:
        self.data_root = Path(data_root)

    @staticmethod
    def parse_run_filename(stem: str) -> tuple[int, int] | None:
        m = EEGMatLoader._fname_re.match(stem + ".mat")
        if not m:
            return None
        return int(m.group("sub")), int(m.group("run"))

    def _normalize_subject(self, subject: str | int) -> tuple[str, str]:
        """Return (folder_name, file_token) e.g. (``S1``, ``S01``)."""
        s = str(subject).strip().upper()
        if s.startswith("S"):
            s = s[1:]

        n = int(s)
        return f"S{n}", f"S{n:02d}"

    def resolve_run_path(self, subject: str | int, run: int) -> Path:
        folder, token = self._normalize_subject(subject)
        path = self.data_root / folder / f"ME_{token}_r{int(run):02d}.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load_run(
        self,
        subject: str | int,
        run: int,
        *,
        sfreq: float,
        apply_montage: bool = True,
        scale_eeg_eog_uv_to_v: bool = True,
    ) -> RawArray:
        """Load one run and return an :class:`mne.io.RawArray`.

        Parameters
        ----------
        subject
            Subject folder id (``1``, ``\"S1\"``, ``\"1\"`` → ``data/S1/``).
        run
            Run number (1 → ``..._r01.mat``).
        sfreq
            Sampling frequency in Hz. These trimmed exports do not contain
            ``EEG.srate``; you must set this from your study metadata.
        apply_montage
            If True, set a digitization montage for channels that define
            non-empty ``X/Y/Z`` in ``chanlocs`` (EEG cap channels).
        scale_eeg_eog_uv_to_v
            If True, scale ``eeg`` and ``eog`` channels by ``1e-6`` so data are
            in volts, as expected by MNE. Non-EEG channels (``misc``) are left
            unchanged.
        """
        path = self.resolve_run_path(subject, run)
        mat = loadmat(
            path,
            struct_as_record=False,
            squeeze_me=True,
            simplify_cells=False,
        )
        eeg = mat["EEG"]
        data = np.asarray(eeg.data, dtype=np.float64)
        locs = eeg.chanlocs

        ch_names = [_chan_label(ch) for ch in locs]
        ch_types = [_infer_ch_type(n) for n in ch_names]

        if scale_eeg_eog_uv_to_v:
            for i, ct in enumerate(ch_types):
                if ct in ("eeg", "eog"):
                    data[i] = data[i] * 1e-6

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = RawArray(data, info, verbose=False)

        events = getattr(eeg, "events", None)
        raw.set_annotations(_events_to_annotations(events, sfreq))

        if apply_montage:
            ch_pos: dict[str, np.ndarray] = {}
            for name, ch in zip(ch_names, locs, strict=True):
                xyz = _xyz_mm(ch)
                if xyz is None:
                    continue
                x_mm, y_mm, z_mm = xyz
                ch_pos[name] = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0

            if ch_pos:
                montage = mne.channels.make_dig_montage(
                    ch_pos=ch_pos, coord_frame="head"
                )
                raw.set_montage(montage, on_missing="ignore")

        return raw
