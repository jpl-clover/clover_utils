import os

from generate_splits import (
    read_filenames_and_labels,
    repo_root,
    resample_fraction_with_preserved_distribution,
    write_filenames_and_labels,
)


def split_and_save(filenames, labels, splits, repo_root, augmented=False):
    # choose only train files
    filenames, labels, splits = zip(
        *[(f, l, s) for f, l, s in zip(filenames, labels, splits) if s == "train"]
    )
    augmented = "_augmented" if augmented else ""
    for seed in range(0, 10):
        out_dir = os.path.join(
            repo_root, "data", f"hirise{augmented}", f"random-cumulative-split-{seed}"
        )
        for pct in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            f_sample, l_sample = resample_fraction_with_preserved_distribution(
                filenames, labels, fraction=pct, seed=seed
            )
            out_file = os.path.join(out_dir, f"{int(100*pct)}pctTrain.txt")
            write_filenames_and_labels(
                f_sample, l_sample, out_file, ["train"] * len(f_sample)
            )
            print(f"Saved {len(f_sample)} images to {out_file}")


if __name__ == "__main__":
    hirise_root = "~/Documents/CLOVER/data/hirise-map-proj-v3_2/"
    file_map = os.path.join(hirise_root, "labels-map-proj_v3_2_train_val_test.txt")
    out_dir = os.path.join(repo_root, "data", "hirise")
    filenames, labels, splits = read_filenames_and_labels(file_map)

    # print(f"Found {len(filenames)} files")
    # classes, freq = np.unique( [int(l) for l in labels] , return_counts=True)
    # [print(f"{c}: {f}") for c, f in zip(classes, freq)]
    # split, freq = np.unique(splits, return_counts=True)
    # [print(f"{s}: {f}") for s, f in zip(split, freq)]

    # Perform stratified split on full training set with augmentations
    split_and_save(filenames, labels, splits, repo_root, augmented=True)

    # Perform stratified split on training set WITHOUT augmentations
    print(f"Filtering for unaugmented files")
    pattern = ["-fh.jpg", "-fv.jpg", "-r90.jpg", "-r180.jpg", "-r270.jpg", "-brt.jpg"]
    augmented = [any(p in f for p in pattern) for f in filenames]
    filenames, labels, splits = zip(
        *[
            (f, l, s)
            for f, l, s, a in zip(filenames, labels, splits, augmented)
            if not a
        ]
    )
    split_and_save(filenames, labels, splits, repo_root, augmented=False)
