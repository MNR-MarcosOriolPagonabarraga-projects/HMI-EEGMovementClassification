import mne
import numpy as np
from scipy.io.matlab import mat_struct
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from src.config import CLASS_NAMES


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

def extract_dl_features(model, data_loader, device, is_dual_branch=False):
    """Extracts pre-softmax features using a forward hook on the final linear layer."""
    features = []
    labels_list = [] 
    
    # Locate the final linear layer dynamically (assumes model.classifier exists)
    # If your classifier is nn.Sequential(nn.Flatten(), nn.Linear()), this grabs the Linear layer.
    final_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            final_layer = module
            
    if final_layer is None:
        raise ValueError("Could not find an nn.Linear layer in the model to hook into.")

    # Define the hook to grab the input to the linear layer (the latent space)
    def hook_fn(module, input, output):
        features.append(input[0].detach().cpu().numpy())

    hook = final_layer.register_forward_hook(hook_fn)
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            if is_dual_branch:
                x1, x2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                model(x1, x2)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                model(x)
            labels_list.extend(y.cpu().numpy())

    hook.remove() # Clean up the hook
    return np.concatenate(features, axis=0), np.array(labels_list)


def plot_and_save_latent_space(features, labels, title, save_path, is_csp=False):
    """Generates a highly styled, minimalistic journal-quality scatter plot."""
    if is_csp:
        # CSP is already spatial, just take the first two components
        Z_2d = features[:, :2]
        x_label, y_label = "CSP Component 1", "CSP Component 2"
    else:
        # Deep learning models need t-SNE dimensionality reduction
        print(f"    Running t-SNE on feature space of shape {features.shape}...")
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        Z_2d = tsne.fit_transform(features)
        x_label, y_label = "t-SNE Dimension 1", "t-SNE Dimension 2"

    fig, ax = plt.subplots(figsize=(5, 5))
    scientific_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for k, class_name in enumerate(CLASS_NAMES):
        mask = (labels == k)
        ax.scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            s=25, alpha=0.65, c=[scientific_colors[k]], 
            label=class_name, edgecolors="none"
        )

    # Minimalist Styling
    ax.set_title(title, fontsize=12, fontweight='medium', pad=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(loc='best', frameon=False, fontsize=10)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved latent visualization to {save_path}")