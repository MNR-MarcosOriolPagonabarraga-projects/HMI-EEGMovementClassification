

from src.load_data import EEGMatLoader
from src.utils import (
    discover_subject_ids,
    build_pooled_arrays,
    stratified_mix_split,
    save_compressed_npz,
)
from src.config import (
    SFREQ, 
    TEST_SIZE, TRAIN_SIZE,
    DATA_ROOT, TRAIN_OUTPUT, TEST_OUTPUT,
)




def main() -> None:
    loader = EEGMatLoader(data_root=DATA_ROOT)
    subjects = discover_subject_ids(DATA_ROOT)

    print(
        f"[build_dataset] subjects={len(subjects)}, "
        f"data_root={DATA_ROOT}\n\tids={subjects}"
    )

    X, y, ch_names = build_pooled_arrays(loader, sfreq=SFREQ)

    X_train, X_test, y_train, y_test = stratified_mix_split(
        X,
        y,
        test_fraction=TEST_SIZE,
        random_state=42,
    )

    train_frac_real = len(y_train) / len(y)
    test_frac_real = len(y_test) / len(y)
    print(
        "[build_dataset] split sizes:"
        f" train {len(y_train)} ({train_frac_real:.3%}),"
        f" test {len(y_test)} ({test_frac_real:.3%});"
        f" target {TRAIN_SIZE:.0%}/{TEST_SIZE:.0%}"
    )

    save_compressed_npz(
        TRAIN_OUTPUT,
        X=X_train,
        y=y_train,
        sfreq=SFREQ,
        ch_names=ch_names,
        split_tag="train",
    )
    save_compressed_npz(
        TEST_OUTPUT,
        X=X_test,
        y=y_test,
        sfreq=SFREQ,
        ch_names=ch_names,
        split_tag="test",
    )

    print(
        f"[build_dataset] wrote {TRAIN_OUTPUT} and {TEST_OUTPUT} "
        f"({len(ch_names)} EEG channels, sfreq={SFREQ} Hz)."
    )


if __name__ == "__main__":
    main()
