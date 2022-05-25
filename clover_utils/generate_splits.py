import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

repo_root = Path(os.path.realpath(__file__)).parent.parent

def read_filenames_and_labels(fp_src):
    """
    Simple file reader utility
    """
    with open(fp_src, "r") as f:
        lines = f.readlines()
    filenames = []
    labels = []
    others = defaultdict(list)
    for line in lines:
        tokens = line.strip().split(" ")
        filenames.append(tokens[0])
        labels.append(tokens[1])
        for n, i in enumerate(range(2, len(tokens))):
            others[n].append(tokens[i])
    if len(others) > 0:
        return filenames, labels, *[others[i] for i in range(len(others))]
    else:
        return filenames, labels


def write_filenames_and_labels(filenames, labels, fn, *args):
    """
    Simple file writer utility
    """
    Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
    with open(fn, "w") as f:
        for i in range(len(filenames)):
            if len(args) > 0:
                suffix = "\t".join([str(args[j][i]) for j in range(len(args))]).strip()
                f.write(f"{filenames[i]}\t{labels[i]}\t{suffix}\n")
            else:
                f.write(f"{filenames[i].strip()} {labels[i].strip()}\n")


def resample_fraction_with_preserved_distribution(
    filenames, labels, fraction=1.0, seed=0
):
    # set random seed to ensure we get the same results every time
    random.seed(seed)
    np.random.seed(seed)

    n = math.floor(len(filenames) * fraction)

    # What is the initial label distribution
    values, counts = np.unique( [int(l) for l in labels] , return_counts=True)

    # How many do we need from each?
    desired_counts = [ max(1, math.floor(c * fraction)) for c in counts ]

    # Where to find examples of each label
    label_indices = [ [] for _ in range(len(values)) ]
    for i, label in enumerate(labels):
        label_indices[int(label)].append(i)

    # Get the final indices based on the desired amounts, using random shuffling
    indices = [ [] for _ in range(len(values)) ]
    for i, label_class_list in enumerate(label_indices):
        indices[i].extend(random.sample(label_class_list, desired_counts[i]))

    # Turn those into filenames and labels
    flat_indices = [item for sublist in indices for item in sublist]
    sampled_filenames = [ filenames[i] for i in flat_indices ]
    sampled_labels = [ labels[i] for i in flat_indices ]

    return sampled_filenames, sampled_labels
