from pathlib import Path
import mne
import numpy as np
from mne.io import RawArray
from scipy.io import loadmat

from src.utils import _chan_label, _infer_ch_type, _xyz_mm, _events_to_annotations

class EEGMatLoader:
    def __init__(
        self, data_root: str | Path = "data", *, channels: list[str] | None = None
    ) -> None:
        self.data_root = Path(data_root)
        self.channels = channels

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

    @staticmethod
    def load_mat_file(
        path: str | Path,
        *,
        sfreq: float,
        apply_montage: bool = True,
        scale_eeg_eog_uv_to_v: bool = True,
        channel_subset: list[str] | None = None,
    ) -> RawArray:
        """Load a MATLAB ``EEGLAB`` export (``.mat``) from disk.

        Parameters
        ----------
        path
            Path to the ``*.mat`` file produced by ``ME_*`` EEGLAB export.
        sfreq
            Sampling frequency (Hz).
        channel_subset
            If provided, restricts to these channel labels *after* load.
        """
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        mat = loadmat(
            path.as_posix(),
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

        if channel_subset is not None:
            raw.pick(channel_subset)

        return raw

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
        raw = EEGMatLoader.load_mat_file(
            path,
            sfreq=sfreq,
            apply_montage=apply_montage,
            scale_eeg_eog_uv_to_v=scale_eeg_eog_uv_to_v,
        )
        # Only keep requested channels, if specified
        if self.channels is not None:
            raw.pick(self.channels)

        return raw
