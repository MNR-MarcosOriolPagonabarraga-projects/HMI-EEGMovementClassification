from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import json
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CLASS_NAMES,
    CSP_N_COMPONENTS,
    CSP_REG,
    DATA_ROOT,
    EVENT_ID,
    MOTOR_CHANNELS,
    REJECT_EEG_UV,
    SFREQ,
    TEST_SIZE,
    TUNE_BANDS_SPEC,
    TUNE_OUTPUT_JSON,
    TUNE_RANDOM_STATE,
    TUNE_WINDOWS_SPEC,
)
from src.load_data import EEGMatLoader
from src.pipeline import EEGPreprocessor

_RUN_STEM_RE = re.compile(r"ME_S(?P<sub>\d+)_r(?P<run>\d+)\.mat$", re.IGNORECASE)


@dataclass
class TrialResult:
    epoch_tmin: float
    epoch_tmax: float
    band_l_hz: float
    band_h_hz: float
    n_trials: int
    n_train: int
    n_test: int
    train_acc: float
    test_acc: float
    seconds: float


def _subject_folder_and_id(subject: str | int) -> tuple[str, int]:
    s = str(subject).strip().upper()
    if s.startswith("S"):
        s = s[1:]
    n = int(s)
    return f"S{n}", n


def _parse_run_stem(stem: str) -> tuple[int, int] | None:
    m = _RUN_STEM_RE.match(stem + ".mat")
    if m is None:
        return None
    return int(m.group("sub")), int(m.group("run"))


def _discover_subject_ids(original_root: Path) -> list[int]:
    ids: list[int] = []
    if not original_root.is_dir():
        raise FileNotFoundError(f"Expected raw data directory at {original_root}")

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
        raise FileNotFoundError(f"No subject folders with ME_*.mat under {original_root}")
    return ids


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


def _parse_pairs(spec: str) -> list[tuple[float, float]]:
    """Parse ``"a,b;c,d"`` into [(a,b), (c,d)]."""
    out: list[tuple[float, float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.replace(" ", "").split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected 'low,high' pair, got {chunk!r}")
        out.append((float(parts[0]), float(parts[1])))
    if not out:
        raise ValueError(f"No pairs parsed from {spec!r}")
    return out


def _make_epochs(
    raw: mne.io.BaseRaw,
    *,
    tmin: float,
    tmax: float,
) -> mne.Epochs:
    events, event_id_sel = mne.events_from_annotations(raw, event_id=EVENT_ID)
    reject = dict(eeg=REJECT_EEG_UV * 1e-6)
    # baseline=(None, 0) needs pre-stimulus samples; at tmin>=0 that collapses to one sample → ValueError
    baseline = (None, 0) if tmin < 0 else None
    epochs = mne.Epochs(
        raw,
        events,
        event_id_sel,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks="eeg",
        reject=reject,
        preload=True,
        verbose=False,
    )
    epochs.drop_bad()
    return epochs


def _epochs_to_xy(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    # float32 + view from preloaded Epochs (same idea as build_dataset._epochs_to_arrays)
    X = epochs.get_data(copy=False).astype(np.float32, copy=False)
    y = epochs.events[:, 2].astype(np.int64, copy=False)
    return X, y


def _all_run_specs(
    loader: EEGMatLoader, subject_ids: Sequence[int]
) -> list[tuple[int, int]]:
    return [
        (sid, run)
        for sid in subject_ids
        for run in _run_indices_for_subject(loader, sid)
    ]


def _collect_epoch_chunks_streaming(
    loader: EEGMatLoader,
    run_specs: Sequence[tuple[int, int]],
    preprocessor_before_epoch: EEGPreprocessor | None,
    ch_names_ref: list[str] | None,
    *,
    windows: Sequence[tuple[float, float]],
    l_freq: float,
    h_freq: float,
    on_run_processed: Callable[[], None] | None = None,
) -> tuple[list[str], list[list[tuple[np.ndarray, np.ndarray]]]]:
    """One continuous run in RAM at a time: notch → band-pass → epoch each window.

    Avoids keeping every subject/run :class:`~mne.io.Raw` preloaded (and a full duplicate per band).
    MATLAB + notch repeat once per frequency band (same order of work as before, without holding
    two copies of all continuous data). Caller should concatenate one window at a time so peak
    trial-array memory stays closer to a single window's stacked trials.
    """
    chunks_per_window: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in windows]
    ref = ch_names_ref

    for sid, run in run_specs:
        raw = loader.load_run(sid, run)
        raw.pick("eeg", verbose=False)
        if preprocessor_before_epoch is not None:
            raw = preprocessor_before_epoch.process(raw)
        raw.load_data()
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            picks="eeg",
            phase="minimum",
            verbose=False,
        )
        names = raw.ch_names.copy()
        if ref is None:
            ref = names
        elif names != ref:
            raise RuntimeError(
                f"Channel order mismatch: expected {ref}, got {names} "
                f"(subject {sid} run {run})"
            )

        for wi, (tmin, tmax) in enumerate(windows):
            epochs = _make_epochs(raw, tmin=tmin, tmax=tmax)
            if len(epochs) == 0:
                continue
            if epochs.ch_names != ref:
                raise RuntimeError(
                    f"Channel mismatch: expected {ref}, got {epochs.ch_names}"
                )
            xi, yi = _epochs_to_xy(epochs)
            chunks_per_window[wi].append((xi, yi))

        del raw
        if on_run_processed is not None:
            on_run_processed()

    if ref is None:
        raise RuntimeError("No runs processed.")

    return ref, chunks_per_window


def _build_model(*, n_components: int, reg: float | None) -> Pipeline:
    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=n_components,
                    reg=reg,
                    log=True,
                    norm_trace=False,
                ),
            ),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )


def _fit_score(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
) -> tuple[float, float, int, int, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )
    n_ch = X_train.shape[1]
    n_comp = min(CSP_N_COMPONENTS, n_ch)
    model = _build_model(n_components=max(1, n_comp), reg=CSP_REG)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    return train_acc, test_acc, len(y_train), len(y_test), y_test, y_test_pred


def _float_slug(x: float) -> str:
    """Stable filesystem token for floats (no minus/dot ambiguity)."""
    return f"{x:g}".replace("-", "m").replace(".", "p")


def _confusion_matrix_stem(
    *,
    band_l_hz: float,
    band_h_hz: float,
    epoch_tmin: float,
    epoch_tmax: float,
    random_state: int,
    n_components: int,
    csp_reg: float | None,
    test_size: float,
    sfreq: float,
    reject_uv: float,
) -> str:
    reg_s = "none" if csp_reg is None else _float_slug(float(csp_reg))
    return (
        f"csp_cm_bp{_float_slug(band_l_hz)}-{_float_slug(band_h_hz)}Hz_"
        f"win{_float_slug(epoch_tmin)}-{_float_slug(epoch_tmax)}s_"
        f"rs{random_state}_csp{n_components}_reg{reg_s}_"
        f"nch{len(MOTOR_CHANNELS)}_ts{_float_slug(test_size)}_sf{_float_slug(sfreq)}_rej{_float_slug(reject_uv)}uv"
    )


def _save_confusion_matrix_svg(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
) -> None:
    labels = np.arange(len(CLASS_NAMES), dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASS_NAMES))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _progress_columns() -> tuple:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


def main() -> int:
    mne.set_log_level("ERROR")

    windows = _parse_pairs(TUNE_WINDOWS_SPEC)
    bands = _parse_pairs(TUNE_BANDS_SPEC)

    loader = EEGMatLoader(
        data_root=DATA_ROOT,
        channels=MOTOR_CHANNELS,
        target_sfreq=SFREQ,
    )
    subject_ids = _discover_subject_ids(DATA_ROOT)
    run_specs = _all_run_specs(loader, subject_ids)
    console = Console()
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Same continuous-domain steps as build_dataset for band-pass search: notch only here,
    # then each grid band-pass is applied per run (no duplicate of all runs in memory).
    preprocessor = EEGPreprocessor(
        apply_filter=False,
        apply_notch=True,
        apply_car=False,
        apply_resample=False,
    )

    results: list[TrialResult] = []
    ch_names: list[str] | None = None

    n_runs = len(run_specs)
    n_jobs_epochs = len(bands) * n_runs
    n_jobs_fits = len(bands) * len(windows)

    cols = _progress_columns()
    with Progress(*cols, console=console, expand=True) as progress:
        epoch_task = progress.add_task(
            "[cyan]Epoch extraction[/] (load → notch → band-pass)",
            total=n_jobs_epochs,
        )
        fit_task = progress.add_task(
            "[magenta]CSP + LDA[/] (windows × bands)",
            total=n_jobs_fits,
        )

        for l_f, h_f in bands:
            hz_tag = f"{l_f:g}-{h_f:g} Hz"
            progress.update(
                epoch_task,
                description=f"[cyan]Epoch extraction[/] ({hz_tag})",
            )

            def _advance_epoch() -> None:
                progress.advance(epoch_task)

            ch_names, chunks_per_window = _collect_epoch_chunks_streaming(
                loader,
                run_specs,
                preprocessor,
                ch_names,
                windows=windows,
                l_freq=l_f,
                h_freq=h_f,
                on_run_processed=_advance_epoch,
            )

            for wi, (tmin, tmax) in enumerate(windows):
                chunks = chunks_per_window[wi]
                if tmax <= tmin:
                    chunks.clear()
                    progress.advance(fit_task)
                    continue
                if not chunks:
                    progress.advance(fit_task)
                    continue
                X = np.concatenate([c[0] for c in chunks], axis=0)
                y = np.concatenate([c[1] for c in chunks], axis=0)
                chunks.clear()
                t_w = time.perf_counter()
                if len(np.unique(y)) < 2:
                    del X, y
                    progress.advance(fit_task)
                    continue

                tr_acc, te_acc, n_tr, n_te, y_te, y_te_hat = _fit_score(
                    X, y, random_state=TUNE_RANDOM_STATE
                )
                elapsed = time.perf_counter() - t_w
                results.append(
                    TrialResult(
                        epoch_tmin=tmin,
                        epoch_tmax=tmax,
                        band_l_hz=l_f,
                        band_h_hz=h_f,
                        n_trials=len(y),
                        n_train=n_tr,
                        n_test=n_te,
                        train_acc=tr_acc,
                        test_acc=te_acc,
                        seconds=elapsed,
                    )
                )
                stem = _confusion_matrix_stem(
                    band_l_hz=l_f,
                    band_h_hz=h_f,
                    epoch_tmin=tmin,
                    epoch_tmax=tmax,
                    random_state=TUNE_RANDOM_STATE,
                    n_components=min(CSP_N_COMPONENTS, X.shape[1]),
                    csp_reg=CSP_REG,
                    test_size=TEST_SIZE,
                    sfreq=float(loader.output_sfreq),
                    reject_uv=float(REJECT_EEG_UV),
                )
                svg_path = results_dir / f"{stem}.svg"
                title = (
                    f"test acc={te_acc:.3f}  "
                    f"BP [{l_f:g},{h_f:g}] Hz  "
                    f"epoch [{tmin:g},{tmax:g}] s"
                )
                _save_confusion_matrix_svg(svg_path, y_te, y_te_hat, title=title)
                progress.update(
                    fit_task,
                    description=(
                        f"[magenta]CSP + LDA[/] [{hz_tag}] "
                        f"n={len(y)} test={te_acc:.3f}"
                    ),
                )
                del X, y
                progress.advance(fit_task)

            del chunks_per_window

    if not results:
        console.print(
            Panel.fit("[red]No successful grid runs[/] (no trials or single-class).", title="CSP tuning")
        )
        return 1

    results.sort(key=lambda r: r.test_acc, reverse=True)
    best = results[0]
    summary = (
        f"[bold]Best test accuracy[/] [green]{best.test_acc:.4f}[/]\n"
        f"[dim]train[/] {best.train_acc:.4f}   [dim]trials[/] {best.n_trials}\n\n"
        f"[bold]Band-pass[/] {best.band_l_hz:g}–{best.band_h_hz:g} Hz\n"
        f"[bold]Epoch[/] [{best.epoch_tmin:g}, {best.epoch_tmax:g}] s\n\n"
        f"[dim]{len(subject_ids)} subjects · {n_runs} runs · "
        f"{len(MOTOR_CHANNELS)} ch · {loader.output_sfreq:g} Hz[/]\n\n"
        f"[dim]Confusion matrices ({len(results)}):[/] [cyan]{results_dir}[/]"
    )
    console.print(Panel.fit(summary, title="CSP preprocessing tuning", border_style="green"))

    out_json = TUNE_OUTPUT_JSON
    if out_json is not None:
        dest = out_json if out_json.is_absolute() else ROOT / out_json
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "best": asdict(best),
            "all": [asdict(r) for r in results],
        }
        dest.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Wrote[/] {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
