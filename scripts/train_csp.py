from pathlib import Path

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

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
    return Pipeline(
        [
            ("csp", csp),
            ("lda", clf),
        ]
    )


def main() -> int:
    
    X_train, y_train, train_meta = load_npz_split(TRAIN_OUTPUT)
    X_test, y_test, test_meta = load_npz_split(TEST_OUTPUT)
    n_c = len(CLASS_NAMES)

    print(
        f"[train_csp] train file={train_meta['path']} "
        f"X{X_train.shape} split={train_meta['split']}"
    )
    print(
        f"[train_csp] test  file={test_meta['path']} "
        f"X{X_test.shape} split={test_meta['split']}"
    )
    print(
        f"[train_csp] sfreq={train_meta['sfreq']} Hz, "
        f"{len(train_meta['ch_names'])} channels, {n_c} classes"
    )

    model = build_model(n_components=CSP_N_COMPONENTS, reg=CSP_REG)
    model.fit(X_train, y_train)

    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_hat)
    test_acc = accuracy_score(y_test, y_test_hat)
    print(f"[train_csp] accuracy train={train_acc:.4f} test={test_acc:.4f}")

    labels = list(range(n_c))
    print("\n[train_csp] classification report (test):\n")
    print(
        classification_report(
            y_test,
            y_test_hat,
            labels=labels,
            target_names=list(CLASS_NAMES),
            digits=4,
            zero_division=0,
        )
    )
    print("[train_csp] confusion matrix (test, rows=true, cols=pred):\n")
    print(confusion_matrix(y_test, y_test_hat, labels=labels))

    return 0


if __name__ == "__main__":
    main()
