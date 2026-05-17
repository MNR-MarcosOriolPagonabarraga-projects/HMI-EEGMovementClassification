import math
import torch
from torch.utils.data import Dataset
from pathlib import Path

import mne
import numpy as np
from mne.io import RawArray
from scipy.io import loadmat

from src.config import RECORDING_SFREQ
from src.utils import _chan_label, _infer_ch_type, _xyz_mm, _events_to_annotations


class EEGMatLoader:
    def __init__(
        self,
        data_root: str | Path = "data",
        *,
        channels: list[str] | None = None,
        native_sfreq: float | None = None,
        target_sfreq: float | None = None,
    ) -> None:
        """Parameters
        ----------
        native_sfreq
            Sample rate of the stored continuous data and of event latencies in the
            ``.mat`` file. Defaults to :data:`src.config.RECORDING_SFREQ`.
        target_sfreq
            If set, :meth:`load_run` returns data resampled to this rate (MNE adjusts
            annotation onsets accordingly). If ``None``, no resampling is applied.
        """
        self.data_root = Path(data_root)
        self.channels = channels
        self.native_sfreq = float(RECORDING_SFREQ if native_sfreq is None else native_sfreq)
        self.target_sfreq = None if target_sfreq is None else float(target_sfreq)

    @property
    def output_sfreq(self) -> float:
        """Sampling rate of :meth:`load_run` output (after optional resampling)."""
        if self.target_sfreq is None or math.isclose(
            self.target_sfreq, self.native_sfreq
        ):
            return self.native_sfreq
        return self.target_sfreq

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
        apply_montage: bool = True,
        scale_eeg_eog_uv_to_v: bool = True,
    ) -> RawArray:
        """Load one run and return an :class:`mne.io.RawArray` at :attr:`output_sfreq`.

        Annotations are built at :attr:`native_sfreq`, then resampling (when
        configured) updates event times so epoching stays aligned.
        """
        path = self.resolve_run_path(subject, run)
        raw = EEGMatLoader.load_mat_file(
            path,
            sfreq=self.native_sfreq,
            apply_montage=apply_montage,
            scale_eeg_eog_uv_to_v=scale_eeg_eog_uv_to_v,
        )
        if self.channels is not None:
            raw.pick(self.channels)

        if self.target_sfreq is not None and not math.isclose(
            self.target_sfreq, self.native_sfreq
        ):
            raw.resample(self.target_sfreq, npad="auto", verbose=False)

        return raw


class EEGPsdDataset(Dataset):
    def __init__(self, X_raw, X_psd, y, is_train=False, crop_size=256):
        self.X_raw = torch.from_numpy(X_raw).float()
        self.X_psd = torch.from_numpy(X_psd).float()
        self.y = torch.from_numpy(y).long()
        self.is_train = is_train
        self.crop_size = crop_size
        
        if self.X_raw.ndim == 3:
            self.X_raw = self.X_raw.unsqueeze(1)
        if self.X_psd.ndim == 3:
            self.X_psd = self.X_psd.unsqueeze(1)
            
        self.max_time = self.X_raw.shape[-1] # Should be 321

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raw = self.X_raw[idx]
        
        # --- TEMPORAL CROPPING ---
        if self.is_train:
            # Pick a random start point
            max_start = self.max_time - self.crop_size
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            # Always take the exact center for validation to keep it deterministic
            start = (self.max_time - self.crop_size) // 2
            
        raw_cropped = raw[:, :, start : start + self.crop_size]
        
        return raw_cropped, self.X_psd[idx], self.y[idx]
