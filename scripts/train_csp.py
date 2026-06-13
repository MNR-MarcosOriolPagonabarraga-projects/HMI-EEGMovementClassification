import os
from pathlib import Path
import joblib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.utils import plot_and_save_latent_space

from src.config import (
    CLASS_NAMES,
    CSP_N_COMPONENTS,
    CSP_REG,
    TEST_OUTPUT,
    TRAIN_OUTPUT,
)


def load_npz_split(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed dataset: {path}")
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float64)
    y = np.asarray(z["y"]).astype(np.int64).ravel()
    meta = {
        "sfreq": float(np.asarray(z["sfreq"]).squeeze()),
        "ch_names": [str(x) for x in np.asarray(z["ch_names"], dtype=object).ravel()],
        "split": str(np.asarray(z["split"]).ravel()[0]),
        "path": path,
    }
    return X, y, meta


def build_model(*, n_components: int, reg: float | None) -> Pipeline:
    csp = CSP(
        n_components=n_components,
        reg=reg,
        log=True,
        norm_trace=False,
    )
    clf = LinearDiscriminantAnalysis()
    return Pipeline([("csp", csp), ("lda", clf)])


def save_confusion_matrix(labels, preds, out_dir):
    """Computes, normalizes, and saves the confusion matrix plot."""
    class_labels = ['Rest', 'Elbow', 'Hand', 'Forearm']
    cm = confusion_matrix(labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_percent, annot=True, fmt=".1f", cmap="Blues", 
        xticklabels=class_labels, yticklabels=class_labels, ax=ax,
        cbar_kws={'label': 'Accuracy (%)'}
    )
    ax.set_title('CSP Validation Confusion Matrix')
    ax.set_xlabel('Predicted Target')
    ax.set_ylabel('True Target')
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def main() -> int:
    # 1. Setup Output Directory
    out_dir = Path("models/csp")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, train_meta = load_npz_split(TRAIN_OUTPUT)
    X_test, y_test, test_meta = load_npz_split(TEST_OUTPUT)
    n_c = len(CLASS_NAMES)

    print(f"[train_csp] train file={train_meta['path']} X{X_train.shape} split={train_meta['split']}")
    print(f"[train_csp] test  file={test_meta['path']} X{X_test.shape} split={test_meta['split']}")
    print(f"[train_csp] sfreq={train_meta['sfreq']} Hz, {len(train_meta['ch_names'])} channels, {n_c} classes")

    model = build_model(n_components=CSP_N_COMPONENTS, reg=CSP_REG)
    model.fit(X_train, y_train)

    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_hat)
    test_acc = accuracy_score(y_test, y_test_hat)
    print(f"[train_csp] accuracy train={train_acc:.4f} test={test_acc:.4f}")

    # 2. Save Confusion Matrix
    save_confusion_matrix(y_test, y_test_hat, out_dir)

    # 3. Save Latent Space
    print("\n[train_csp] Generating latent space visualization...")
    Z_test = model.named_steps['csp'].transform(X_test)
    plot_and_save_latent_space(
        features=Z_test, 
        labels=y_test, 
        title="CSP Feature Space (Baseline)", 
        save_path=out_dir / "csp_latent_space.png", 
        is_csp=True
    )

    # 4. Save Model Pipeline
    joblib.dump(model, out_dir / "csp_lda_pipeline.joblib")
    print(f"\n[train_csp] Artifacts and model saved to {out_dir}/")

    return 0

if __name__ == "__main__":
    main()