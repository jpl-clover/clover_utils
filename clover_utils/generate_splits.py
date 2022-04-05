import os
import random
import math
import numpy as np
from pathlib import Path

def read_filenames_and_labels(fp_src):
    """
    Simple file reader utility
    """
    with open(fp_src, "r") as f:
        lines = f.readlines()
    filenames = []
    labels = []
    for line in lines:
        tokens = line.split(" ")
        filenames.append(tokens[0])
        labels.append(tokens[1])
    return filenames, labels

def write_filenames_and_labels(filenames, labels, fn):
    """
    Simple file writer utility
    """
    Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
    with open(fn, "w") as f:
        for i in range(len(filenames)):
            f.write(f"{filenames[i]} {labels[i]}")

def resample_fraction_with_preserved_distribution(filenames, labels, fraction=1.0):
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
